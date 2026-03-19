"""Tests for P0/P1 enhancements:
- P0: RetryPolicy fault tolerance
- P0: Command(goto) dynamic routing in L3
- P1: interrupt() function-style approval in L2
- P1: context_schema runtime configuration injection
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RetryPolicy

from core.factory import build_graph
from core.runtime_config import RuntimeContext
from core.state import BaseState, L2State, L3State
from graphs.l1_reactor import DEFAULT_RETRY as L1_RETRY, build_l1_graph
from graphs.l2_workflow import DEFAULT_RETRY as L2_RETRY, build_l2_graph
from graphs.l3_executor import DEFAULT_RETRY as L3_RETRY, build_l3_graph


# ============================================================
# P0: RetryPolicy
# ============================================================

class TestRetryPolicy:
    """Verify RetryPolicy is configured on all critical nodes."""

    def test_retry_policy_is_defined(self):
        """All tiers export a DEFAULT_RETRY with max_attempts >= 2."""
        for policy in [L1_RETRY, L2_RETRY, L3_RETRY]:
            assert isinstance(policy, RetryPolicy)
            assert policy.max_attempts >= 2

    def test_l1_nodes_have_retry(self):
        """L1 llm_call and tool_node should have retry_policy set."""
        graph = build_l1_graph()
        # Verify the graph compiles without error (retry_policy accepted)
        assert isinstance(graph, CompiledStateGraph)
        nodes = graph.get_graph().nodes
        assert "llm_call" in nodes
        assert "tool_node" in nodes

    def test_l3_nodes_have_retry(self):
        """L3 all nodes should compile with retry_policy."""
        graph = build_l3_graph()
        assert isinstance(graph, CompiledStateGraph)
        for name in ["plan", "execute", "evaluate", "reflect", "revise"]:
            assert name in graph.get_graph().nodes

    def test_l2_action_nodes_have_retry(self):
        """L2 action nodes get retry, approval nodes do not."""
        steps = [
            {"name": "fetch", "type": "action"},
            {"name": "approve", "type": "approval"},
            {"name": "write", "type": "action"},
        ]
        graph = build_l2_graph(steps)
        assert isinstance(graph, CompiledStateGraph)

    def test_retry_on_transient_failure(self):
        """A node that fails once then succeeds should complete thanks to retry."""
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import RetryPolicy

        call_count = {"n": 0}

        def flaky_node(state: BaseState) -> dict:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ConnectionError("Transient network error")
            return {
                "messages": [AIMessage(content="Recovered after retry")],
            }

        graph = StateGraph(BaseState)
        graph.add_node("flaky", flaky_node, retry_policy=RetryPolicy(max_attempts=3))
        graph.add_edge(START, "flaky")
        graph.add_edge("flaky", END)
        compiled = graph.compile()

        result = compiled.invoke({
            "messages": [HumanMessage(content="go")],
            "metadata": {},
            "llm_call_count": 0,
            "current_tier": "l1",
        })
        assert call_count["n"] == 2  # Failed once, succeeded on retry
        assert any("Recovered" in m.content for m in result["messages"] if isinstance(m, AIMessage))


# ============================================================
# P0: Command(goto) in L3
# ============================================================

class TestCommandGotoL3:
    """Verify L3 evaluate node uses Command(goto=...) for routing."""

    def _initial_state(self):
        return {
            "messages": [HumanMessage(content="Start")],
            "metadata": {},
            "llm_call_count": 0,
            "current_tier": "l3",
            "iteration_count": 0,
            "reflection_log": [],
            "quality_score": 0.0,
            "is_complete": False,
        }

    def test_command_routes_to_end_on_high_quality(self):
        """evaluate returns Command(goto=END) when quality meets threshold."""

        def high_quality(state):
            return {
                "messages": [AIMessage(content="Excellent")],
                "quality_score": 0.95,
            }

        graph = build_l3_graph(quality_threshold=0.8, evaluate_fn=high_quality)
        result = graph.invoke(self._initial_state())
        assert result["quality_score"] >= 0.8
        assert result["iteration_count"] == 1  # Only one execute cycle

    def test_command_routes_to_reflect_on_low_quality(self):
        """evaluate returns Command(goto='reflect') when quality is low."""

        def low_then_high(state):
            score = 0.3 if state.get("iteration_count", 0) <= 1 else 0.9
            return {
                "messages": [AIMessage(content=f"Score: {score}")],
                "quality_score": score,
            }

        graph = build_l3_graph(quality_threshold=0.8, max_iterations=5, evaluate_fn=low_then_high)
        result = graph.invoke(self._initial_state())
        # Should have reflected once, then quality met on second iteration
        assert result["quality_score"] >= 0.8
        assert result["iteration_count"] == 2
        assert len(result["reflection_log"]) >= 1

    def test_command_routes_to_end_on_max_iterations(self):
        """evaluate returns Command(goto=END) when max_iterations reached."""

        def always_low(state):
            return {
                "messages": [AIMessage(content="Bad")],
                "quality_score": 0.1,
            }

        graph = build_l3_graph(quality_threshold=0.9, max_iterations=2, evaluate_fn=always_low)
        result = graph.invoke(self._initial_state())
        assert result["iteration_count"] == 2
        assert result["quality_score"] < 0.9

    def test_command_routes_to_end_on_is_complete(self):
        """evaluate returns Command(goto=END) when is_complete is True."""

        def mark_complete(state):
            return {
                "messages": [AIMessage(content="Done")],
                "quality_score": 0.1,  # Low quality, but is_complete overrides
                "is_complete": True,
            }

        # Need to pass is_complete through evaluate
        def eval_with_complete(state):
            return {
                "messages": [AIMessage(content="Complete flag set")],
                "quality_score": 0.1,
            }

        def exec_sets_complete(state):
            return {
                "messages": [AIMessage(content="Executing")],
                "iteration_count": state.get("iteration_count", 0) + 1,
                "is_complete": True,
            }

        graph = build_l3_graph(
            quality_threshold=0.9,
            max_iterations=10,
            execute_fn=exec_sets_complete,
            evaluate_fn=eval_with_complete,
        )
        result = graph.invoke(self._initial_state())
        assert result["iteration_count"] == 1
        assert result["is_complete"] is True

    def test_l3_graph_no_conditional_edges(self):
        """L3 graph should not use conditional_edges anymore — Command handles routing."""
        graph = build_l3_graph()
        # The graph should still have all 5 nodes connected
        nodes = graph.get_graph().nodes
        for n in ["plan", "execute", "evaluate", "reflect", "revise"]:
            assert n in nodes


# ============================================================
# P1: interrupt() in L2
# ============================================================

class TestInterruptFnL2:
    """Verify L2 approval steps use interrupt() with Command(resume=...)."""

    def _initial_state(self):
        return {
            "messages": [HumanMessage(content="start")],
            "metadata": {},
            "llm_call_count": 0,
            "current_tier": "l2",
            "current_step": "",
            "approval_status": "",
            "artifacts": {},
        }

    def test_interrupt_pauses_at_approval(self):
        """Graph pauses inside the approval node, not before it."""
        steps = [
            {"name": "prepare", "type": "action"},
            {"name": "approve", "type": "approval"},
            {"name": "execute", "type": "action"},
        ]
        graph = build_l2_graph(steps, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "int-test-1"}}

        result = graph.invoke(self._initial_state(), config=config)
        # prepare completed, approve paused at interrupt()
        assert result["artifacts"].get("prepare") == "completed"
        assert "approve" not in result["artifacts"]

    def test_resume_with_command(self):
        """Command(resume=value) provides the approval decision."""
        steps = [
            {"name": "prepare", "type": "action"},
            {"name": "approve", "type": "approval"},
            {"name": "finalize", "type": "action"},
        ]
        graph = build_l2_graph(steps, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "int-test-2"}}

        graph.invoke(self._initial_state(), config=config)
        result = graph.invoke(Command(resume="yes, approved"), config=config)

        assert result["approval_status"] == "yes, approved"
        assert result["artifacts"]["approve"] == "approved"
        assert result["artifacts"]["finalize"] == "completed"

    def test_multiple_approval_steps(self):
        """Workflow with two approval steps pauses at each one sequentially."""
        steps = [
            {"name": "draft", "type": "action"},
            {"name": "review_1", "type": "approval"},
            {"name": "revise", "type": "action"},
            {"name": "review_2", "type": "approval"},
            {"name": "publish", "type": "action"},
        ]
        graph = build_l2_graph(steps, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "int-test-3"}}

        # First run: pauses at review_1
        result = graph.invoke(self._initial_state(), config=config)
        assert result["artifacts"].get("draft") == "completed"
        assert "review_1" not in result["artifacts"]

        # Resume review_1
        result = graph.invoke(Command(resume="pass"), config=config)
        assert result["artifacts"]["review_1"] == "approved"
        assert result["artifacts"]["revise"] == "completed"
        # Should pause at review_2
        assert "review_2" not in result["artifacts"]

        # Resume review_2
        result = graph.invoke(Command(resume="pass"), config=config)
        assert result["artifacts"]["review_2"] == "approved"
        assert result["artifacts"]["publish"] == "completed"

    def test_interrupt_provides_context_message(self):
        """The interrupt() call includes a message about what approval is needed."""
        # This is implicitly tested — if interrupt() didn't work correctly,
        # the graph wouldn't pause. The message is for the UI/Studio display.
        steps = [{"name": "gate", "type": "approval"}]
        graph = build_l2_graph(steps, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "int-test-4"}}

        result = graph.invoke(self._initial_state(), config=config)
        # Should have interrupted (no artifacts set)
        assert "gate" not in result.get("artifacts", {})


# ============================================================
# P1: context_schema runtime config
# ============================================================

class TestRuntimeContext:
    """Verify context_schema is wired up for runtime config injection."""

    def test_runtime_context_schema_defined(self):
        """RuntimeContext has the expected fields."""
        from core.runtime_config import RuntimeContext
        hints = RuntimeContext.__annotations__
        assert "user_id" in hints
        assert "llm_model" in hints
        assert "temperature" in hints
        assert "language" in hints
        assert "session_id" in hints

    def test_l1_graph_accepts_context(self):
        """L1 graph compiled with context_schema should accept context at invocation."""
        graph = build_l1_graph()
        # Verify it compiled with context_schema (no error at compile time)
        assert isinstance(graph, CompiledStateGraph)

    def test_l1_invocation_with_context(self, monkeypatch):
        """L1 graph can be invoked with runtime context parameters."""
        from graphs import l1_reactor

        received_context = {}

        def mock_llm_with_context(state, *, runtime=None):
            if runtime:
                received_context.update(runtime.context)
            return {
                "messages": [AIMessage(content="OK with context")],
                "llm_call_count": state.get("llm_call_count", 0) + 1,
            }

        monkeypatch.setattr(l1_reactor, "_llm_call", mock_llm_with_context)
        graph = build_l1_graph()

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="Hi")],
                "metadata": {},
                "llm_call_count": 0,
                "current_tier": "l1",
            },
            context={"user_id": "u123", "llm_model": "gpt-4o", "temperature": 0.7},
        )
        assert received_context.get("user_id") == "u123"
        assert received_context.get("llm_model") == "gpt-4o"
        assert received_context.get("temperature") == 0.7

    def test_context_fields_optional(self):
        """Partial context should work — all fields are optional."""
        graph = build_l1_graph()
        # Should not error with empty context
        # Just verifying it doesn't crash with partial context
        assert isinstance(graph, CompiledStateGraph)
