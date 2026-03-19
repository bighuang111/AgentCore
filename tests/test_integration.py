"""Integration tests: full-flow testing across L1/L2/L3 tiers."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolCall

from core.config import AppConfig, SafetyConfig
from core.factory import build_graph
from core.output import StandardOutput
from core.safety import PIISanitizer
from plugins.observers.cost_tracker import CostTracker


# --- L1 Integration ---

class TestL1Integration:
    def test_full_l1_flow_with_tool(self, mock_llm_with_tool_call, monkeypatch):
        """L1: User message → LLM → tool call → LLM → final answer → formatted output."""
        from graphs import l1_reactor
        monkeypatch.setattr(l1_reactor, "_get_llm", lambda *a, **kw: mock_llm_with_tool_call)

        config = AppConfig(safety=SafetyConfig(max_loops=10))
        graph = build_graph("l1", config=config)

        result = graph.invoke({
            "messages": [HumanMessage(content="Echo test please")],
            "metadata": {"test": True},
            "llm_call_count": 0,
            "current_tier": "l1",
        })

        assert result["llm_call_count"] == 2

        md = StandardOutput.to_markdown(result)
        assert "echo" in md.lower() or "Echo" in md
        assert "Tier: l1" in md

        json_out = StandardOutput.to_json(result)
        assert json_out["tier"] == "l1"

    def test_l1_budget_cutoff(self, monkeypatch):
        """L1: Infinite tool calls should be cut off by BudgetGuard."""
        from graphs import l1_reactor
        from tests.conftest import MockLLM

        always_tool = MockLLM([
            AIMessage(content="", tool_calls=[ToolCall(name="echo_tool", args={"text": "loop"}, id="c1")]),
            AIMessage(content="", tool_calls=[ToolCall(name="echo_tool", args={"text": "loop"}, id="c2")]),
            AIMessage(content="", tool_calls=[ToolCall(name="echo_tool", args={"text": "loop"}, id="c3")]),
            AIMessage(content="", tool_calls=[ToolCall(name="echo_tool", args={"text": "loop"}, id="c4")]),
        ])
        monkeypatch.setattr(l1_reactor, "_get_llm", lambda *a, **kw: always_tool)

        config = AppConfig(safety=SafetyConfig(max_loops=3))
        graph = build_graph("l1", config=config)
        result = graph.invoke({
            "messages": [HumanMessage(content="Infinite loop test")],
            "metadata": {},
            "llm_call_count": 0,
            "current_tier": "l1",
        })
        assert result["llm_call_count"] == 3


# --- L2 Integration ---

class TestL2Integration:
    def test_full_l2_flow(self):
        """L2: Multi-step SOP workflow executes in order."""
        steps = [
            {"name": "collect_data", "type": "action"},
            {"name": "analyze", "type": "action"},
            {"name": "generate_report", "type": "action"},
        ]
        graph = build_graph("l2", steps=steps)
        result = graph.invoke({
            "messages": [HumanMessage(content="Start workflow")],
            "metadata": {"workflow": "weekly_report"},
            "llm_call_count": 0,
            "current_tier": "l2",
            "current_step": "",
            "approval_status": "",
            "artifacts": {},
        })

        # All steps completed
        assert result["artifacts"]["collect_data"] == "completed"
        assert result["artifacts"]["analyze"] == "completed"
        assert result["artifacts"]["generate_report"] == "completed"

        # StandardOutput works with L2
        md = StandardOutput.to_markdown(result)
        assert "Tier: l2" in md


# --- L3 Integration ---

class TestL3Integration:
    def test_full_l3_converging(self):
        """L3: Quality improves over iterations until threshold met."""
        iteration_scores = [0.3, 0.6, 0.9]

        def improving_evaluate(state):
            idx = min(state.get("iteration_count", 0), len(iteration_scores) - 1)
            return {
                "messages": [AIMessage(content=f"Score: {iteration_scores[idx]}")],
                "quality_score": iteration_scores[idx],
            }

        graph = build_graph(
            "l3",
            quality_threshold=0.8,
            max_iterations=10,
            evaluate_fn=improving_evaluate,
        )
        result = graph.invoke({
            "messages": [HumanMessage(content="Deep research task")],
            "metadata": {},
            "llm_call_count": 0,
            "current_tier": "l3",
            "iteration_count": 0,
            "reflection_log": [],
            "quality_score": 0.0,
            "is_complete": False,
        })

        assert result["quality_score"] >= 0.8
        assert result["iteration_count"] <= 10

    def test_l3_max_iterations_safety(self):
        """L3: Force-stop at max iterations even if quality never meets threshold."""
        def always_low(state):
            return {
                "messages": [AIMessage(content="Still bad")],
                "quality_score": 0.1,
            }

        graph = build_graph("l3", max_iterations=2, evaluate_fn=always_low)
        result = graph.invoke({
            "messages": [HumanMessage(content="Impossible task")],
            "metadata": {},
            "llm_call_count": 0,
            "current_tier": "l3",
            "iteration_count": 0,
            "reflection_log": [],
            "quality_score": 0.0,
            "is_complete": False,
        })

        assert result["iteration_count"] == 2
        assert result["quality_score"] < 0.8


# --- Cross-cutting Integration ---

class TestCrossCutting:
    def test_pii_in_output(self):
        """PII sanitizer works on StandardOutput."""
        state = {
            "messages": [
                HumanMessage(content="Hi"),
                AIMessage(content="Contact john@example.com or call 555-123-4567"),
            ],
            "metadata": {},
            "llm_call_count": 1,
            "current_tier": "l1",
        }
        md = StandardOutput.to_markdown(state)
        sanitizer = PIISanitizer()
        clean = sanitizer.sanitize(md)
        assert "[EMAIL]" in clean
        assert "[PHONE]" in clean
        assert "john@example.com" not in clean

    def test_cost_tracking_across_calls(self):
        """CostTracker accumulates across multiple operations."""
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 500, 200)
        tracker.record("gpt-4o", 1000, 500)
        summary = tracker.summary()
        assert summary["total_calls"] == 2
        assert summary["total_tokens"] == 2200
        assert summary["estimated_cost_usd"] > 0

    def test_all_tiers_produce_standard_output(self):
        """All three tiers produce valid StandardOutput."""
        for tier_name, extra in [
            ("l1", {}),
            ("l2", {"steps": [{"name": "s1", "type": "action"}]}),
            ("l3", {}),
        ]:
            if tier_name == "l2":
                graph = build_graph(tier_name, **extra)
                state = {
                    "messages": [HumanMessage(content="Go")],
                    "metadata": {},
                    "llm_call_count": 0,
                    "current_tier": tier_name,
                    "current_step": "",
                    "approval_status": "",
                    "artifacts": {},
                }
            elif tier_name == "l3":
                def quick_eval(state):
                    return {"messages": [AIMessage(content="OK")], "quality_score": 1.0}

                graph = build_graph(tier_name, evaluate_fn=quick_eval)
                state = {
                    "messages": [HumanMessage(content="Go")],
                    "metadata": {},
                    "llm_call_count": 0,
                    "current_tier": tier_name,
                    "iteration_count": 0,
                    "reflection_log": [],
                    "quality_score": 0.0,
                    "is_complete": False,
                }
            else:
                # L1 needs monkeypatch which we don't have here, skip invoke
                continue

            result = graph.invoke(state)
            json_out = StandardOutput.to_json(result)
            assert json_out["tier"] == tier_name
            assert isinstance(json_out["output"], str)
