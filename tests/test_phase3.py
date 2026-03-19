"""Phase 3 tests: L2 workflow with SOP steps and approval gates.

Updated for interrupt() based approval (P1 enhancement).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from core.factory import build_graph
from core.state import L2State
from graphs.l2_workflow import build_l2_graph


_L2_INITIAL = {
    "messages": [HumanMessage(content="start")],
    "metadata": {},
    "llm_call_count": 0,
    "current_tier": "l2",
    "current_step": "",
    "approval_status": "",
    "artifacts": {},
}


class TestL2State:
    def test_has_current_step(self):
        assert "current_step" in L2State.__annotations__

    def test_has_approval_status(self):
        assert "approval_status" in L2State.__annotations__

    def test_has_artifacts(self):
        assert "artifacts" in L2State.__annotations__


class TestL2Workflow:
    def test_build_l2_returns_compiled_graph(self):
        steps = [
            {"name": "gather", "type": "action"},
            {"name": "process", "type": "action"},
        ]
        graph = build_l2_graph(steps)
        assert isinstance(graph, CompiledStateGraph)

    def test_empty_steps_raises(self):
        with pytest.raises(ValueError, match="at least one step"):
            build_l2_graph([])

    def test_linear_execution_order(self):
        steps = [
            {"name": "step_a", "type": "action"},
            {"name": "step_b", "type": "action"},
            {"name": "step_c", "type": "action"},
        ]
        graph = build_l2_graph(steps)
        result = graph.invoke(dict(_L2_INITIAL))
        assert result["artifacts"]["step_a"] == "completed"
        assert result["artifacts"]["step_b"] == "completed"
        assert result["artifacts"]["step_c"] == "completed"
        assert result["current_step"] == "step_c"

    def test_approval_node_interrupts_with_interrupt_fn(self):
        """Approval nodes use interrupt() — pauses inside the node, resumes with Command."""
        steps = [
            {"name": "gather", "type": "action"},
            {"name": "review", "type": "approval"},
            {"name": "publish", "type": "action"},
        ]
        graph = build_l2_graph(steps, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-approval-1"}}

        # First invoke: should execute "gather", then pause inside "review" at interrupt()
        result = graph.invoke(dict(_L2_INITIAL), config=config)
        assert result["artifacts"].get("gather") == "completed"
        assert "review" not in result["artifacts"]

        # Resume with Command(resume=...) — the approval value
        result = graph.invoke(Command(resume="approved"), config=config)
        assert result["artifacts"]["review"] == "approved"
        assert result["approval_status"] == "approved"
        assert result["artifacts"]["publish"] == "completed"

    def test_approval_resume_value_propagates(self):
        """The resume value from Command(resume=...) becomes the approval_status."""
        steps = [
            {"name": "check", "type": "action"},
            {"name": "sign_off", "type": "approval"},
        ]
        graph = build_l2_graph(steps, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-approval-2"}}

        graph.invoke(dict(_L2_INITIAL), config=config)
        result = graph.invoke(Command(resume="rejected"), config=config)
        assert result["approval_status"] == "rejected"


class TestFactoryL2:
    def test_factory_builds_l2(self):
        steps = [{"name": "s1", "type": "action"}, {"name": "s2", "type": "action"}]
        graph = build_graph("l2", steps=steps)
        assert isinstance(graph, CompiledStateGraph)

    def test_l1_and_l2_coexist(self):
        l1 = build_graph("l1")
        l2 = build_graph("l2", steps=[{"name": "s1", "type": "action"}])
        assert "llm_call" in l1.get_graph().nodes
        assert "s1" in l2.get_graph().nodes
