"""Phase 2 tests: safety gateway, config-driven graph building."""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolCall

from core.config import AppConfig, SafetyConfig
from core.factory import build_graph
from core.safety import BudgetGuard


# --- Config tests ---

class TestAppConfig:
    def test_default_config(self):
        config = AppConfig()
        assert config.default_llm == "openai:gpt-4o-mini"
        assert config.safety.max_loops == 10
        assert "l1" in config.enabled_tiers

    def test_from_yaml(self):
        config = AppConfig.from_yaml("user_config.yaml")
        assert config.safety.max_loops == 10
        assert "echo_tool" in config.enabled_tools

    def test_from_yaml_missing_file(self):
        config = AppConfig.from_yaml("nonexistent.yaml")
        assert config.default_llm == "openai:gpt-4o-mini"

    def test_custom_yaml(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            default_llm: "openai:gpt-4o"
            safety:
              max_loops: 3
              max_tokens: 2048
            enabled_tiers:
              - l1
              - l2
            enabled_tools:
              - echo_tool
        """)
        p = tmp_path / "test_config.yaml"
        p.write_text(yaml_content)
        config = AppConfig.from_yaml(p)
        assert config.safety.max_loops == 3
        assert config.default_llm == "openai:gpt-4o"
        assert "l2" in config.enabled_tiers


# --- BudgetGuard tests ---

class TestBudgetGuard:
    def test_allows_tool_call_within_budget(self):
        guard = BudgetGuard(max_loops=5)
        state = {
            "messages": [AIMessage(content="", tool_calls=[
                ToolCall(name="echo_tool", args={"text": "hi"}, id="1")
            ])],
            "llm_call_count": 2,
            "metadata": {},
            "current_tier": "l1",
        }
        assert guard.should_continue(state) == "tool_node"

    def test_blocks_at_max_loops(self):
        guard = BudgetGuard(max_loops=2)
        state = {
            "messages": [AIMessage(content="", tool_calls=[
                ToolCall(name="echo_tool", args={"text": "hi"}, id="1")
            ])],
            "llm_call_count": 2,
            "metadata": {},
            "current_tier": "l1",
        }
        assert guard.should_continue(state) == "__end__"

    def test_ends_without_tool_calls(self):
        guard = BudgetGuard(max_loops=10)
        state = {
            "messages": [AIMessage(content="Done.")],
            "llm_call_count": 1,
            "metadata": {},
            "current_tier": "l1",
        }
        assert guard.should_continue(state) == "__end__"


# --- Factory with config tests ---

class TestFactoryWithConfig:
    def test_factory_uses_config_tools(self):
        config = AppConfig(enabled_tools=["echo_tool"])
        graph = build_graph("l1", config=config)
        assert "llm_call" in graph.get_graph().nodes

    def test_max_loops_truncation(self, monkeypatch):
        """With max_loops=2, graph should stop after 2 LLM calls even if tool calls remain."""
        from graphs import l1_reactor
        from tests.conftest import MockLLM

        # Mock LLM that always returns tool calls
        always_tool = MockLLM([
            AIMessage(content="", tool_calls=[ToolCall(name="echo_tool", args={"text": "loop"}, id="c1")]),
            AIMessage(content="", tool_calls=[ToolCall(name="echo_tool", args={"text": "loop"}, id="c2")]),
            AIMessage(content="", tool_calls=[ToolCall(name="echo_tool", args={"text": "loop"}, id="c3")]),
        ])
        monkeypatch.setattr(l1_reactor, "_get_llm", lambda *a, **kw: always_tool)

        config = AppConfig(safety=SafetyConfig(max_loops=2))
        graph = build_graph("l1", config=config)
        result = graph.invoke({
            "messages": [HumanMessage(content="Loop forever")],
            "metadata": {},
            "llm_call_count": 0,
            "current_tier": "l1",
        })
        # Should have stopped at max_loops=2
        assert result["llm_call_count"] == 2
