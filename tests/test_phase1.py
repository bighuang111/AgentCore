"""Phase 1 tests: skeleton, BaseState, L1 reactor."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langgraph.graph.state import CompiledStateGraph

from core.factory import build_graph
from core.state import BaseState
from plugins.tools.basic_tools import echo_tool, current_time, get_basic_tools


# --- BaseState tests ---

class TestBaseState:
    def test_has_messages_field(self):
        annotations = BaseState.__annotations__
        assert "messages" in annotations

    def test_has_metadata_field(self):
        assert "metadata" in BaseState.__annotations__

    def test_has_llm_call_count_field(self):
        assert "llm_call_count" in BaseState.__annotations__

    def test_has_current_tier_field(self):
        assert "current_tier" in BaseState.__annotations__


# --- Factory tests ---

class TestFactory:
    def test_build_l1_returns_compiled_graph(self):
        graph = build_graph("l1")
        assert isinstance(graph, CompiledStateGraph)

    def test_build_unknown_tier_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown tier"):
            build_graph("l99")


# --- L1 Graph structure tests ---

class TestL1GraphStructure:
    def test_has_llm_call_node(self):
        graph = build_graph("l1")
        assert "llm_call" in graph.get_graph().nodes

    def test_has_tool_node(self):
        graph = build_graph("l1")
        assert "tool_node" in graph.get_graph().nodes


# --- Basic tools tests ---

class TestBasicTools:
    def test_echo_tool(self):
        result = echo_tool.invoke({"text": "hello"})
        assert result == "Echo: hello"

    def test_current_time(self):
        result = current_time.invoke({})
        assert "T" in result  # ISO format contains T

    def test_get_basic_tools_returns_list(self):
        tools = get_basic_tools()
        assert len(tools) == 2


# --- L1 full invocation with mock LLM ---

class TestL1Invocation:
    def test_simple_response(self, mock_llm_simple, monkeypatch):
        """Mock LLM returns text without tool calls -> graph ends."""
        from graphs import l1_reactor
        monkeypatch.setattr(l1_reactor, "_get_llm", lambda *a, **kw: mock_llm_simple)

        graph = build_graph("l1")
        result = graph.invoke({
            "messages": [HumanMessage(content="Hi")],
            "metadata": {},
            "llm_call_count": 0,
            "current_tier": "l1",
        })
        assert any("mock LLM" in m.content for m in result["messages"] if isinstance(m, AIMessage))

    def test_tool_call_then_response(self, mock_llm_with_tool_call, monkeypatch):
        """Mock LLM calls echo_tool, then returns final answer."""
        from graphs import l1_reactor
        monkeypatch.setattr(l1_reactor, "_get_llm", lambda *a, **kw: mock_llm_with_tool_call)

        graph = build_graph("l1")
        result = graph.invoke({
            "messages": [HumanMessage(content="Echo test")],
            "metadata": {},
            "llm_call_count": 0,
            "current_tier": "l1",
        })
        # Should have called LLM twice (tool call + final)
        assert result["llm_call_count"] == 2
