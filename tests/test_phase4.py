"""Phase 4 tests: L3 autonomous executor with reflection loop."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from core.factory import build_graph
from core.state import L3State
from graphs.l3_executor import build_l3_graph


def _initial_l3_state() -> dict:
    return {
        "messages": [HumanMessage(content="Start L3 task")],
        "metadata": {},
        "llm_call_count": 0,
        "current_tier": "l3",
        "iteration_count": 0,
        "reflection_log": [],
        "quality_score": 0.0,
        "is_complete": False,
    }


class TestL3State:
    def test_has_iteration_count(self):
        assert "iteration_count" in L3State.__annotations__

    def test_has_reflection_log(self):
        assert "reflection_log" in L3State.__annotations__

    def test_has_quality_score(self):
        assert "quality_score" in L3State.__annotations__

    def test_has_is_complete(self):
        assert "is_complete" in L3State.__annotations__


class TestL3QualityTermination:
    def test_stops_when_quality_met(self):
        """Graph stops when evaluate returns quality >= threshold."""

        call_count = {"n": 0}

        def evaluate_high(state):
            call_count["n"] += 1
            return {
                "messages": [AIMessage(content="Quality OK")],
                "quality_score": 0.9,  # Above 0.8 threshold
            }

        graph = build_l3_graph(
            quality_threshold=0.8,
            max_iterations=10,
            evaluate_fn=evaluate_high,
        )
        result = graph.invoke(_initial_l3_state())
        # Should have stopped after first evaluate (quality met)
        assert result["quality_score"] >= 0.8
        assert result["iteration_count"] == 1


class TestL3MaxIterations:
    def test_force_exit_at_max_iterations(self):
        """Graph stops when iteration_count >= max_iterations."""

        def evaluate_low(state):
            return {
                "messages": [AIMessage(content="Quality low")],
                "quality_score": 0.1,  # Always below threshold
            }

        graph = build_l3_graph(
            quality_threshold=0.8,
            max_iterations=3,
            evaluate_fn=evaluate_low,
        )
        result = graph.invoke(_initial_l3_state())
        assert result["iteration_count"] == 3
        assert result["quality_score"] < 0.8
        assert len(result["reflection_log"]) > 0


class TestL3ReflectionLoop:
    def test_reflection_log_populated(self):
        """Reflection log should accumulate entries."""

        def evaluate_low(state):
            return {
                "messages": [AIMessage(content="Low quality")],
                "quality_score": 0.1,
            }

        graph = build_l3_graph(max_iterations=2, evaluate_fn=evaluate_low)
        result = graph.invoke(_initial_l3_state())
        assert len(result["reflection_log"]) >= 1


class TestAllTiersCoexist:
    def test_three_tiers(self):
        l1 = build_graph("l1")
        l2 = build_graph("l2", steps=[{"name": "s1", "type": "action"}])
        l3 = build_graph("l3")
        assert isinstance(l1, CompiledStateGraph)
        assert isinstance(l2, CompiledStateGraph)
        assert isinstance(l3, CompiledStateGraph)

    def test_l3_graph_has_expected_nodes(self):
        graph = build_l3_graph()
        nodes = graph.get_graph().nodes
        for name in ["plan", "execute", "evaluate", "reflect", "revise"]:
            assert name in nodes, f"Missing node: {name}"
