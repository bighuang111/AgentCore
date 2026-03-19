"""Shared test fixtures with Mock LLM support."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall


class MockLLM:
    """A mock LLM that returns predefined responses.

    Supports tool binding and can simulate tool calls.
    Uses shared call counter so bind_tools clones stay in sync.
    """

    def __init__(self, responses: list[AIMessage] | None = None, _shared_count: list | None = None):
        self._responses = list(responses or [])
        # Shared mutable counter so bind_tools clones share the sequence
        self._shared_count = _shared_count if _shared_count is not None else [0]
        self._bound_tools: list = []

    def bind_tools(self, tools: list) -> "MockLLM":
        clone = MockLLM(self._responses, _shared_count=self._shared_count)
        clone._bound_tools = tools
        return clone

    def invoke(self, messages: Any, **kwargs) -> AIMessage:
        idx = self._shared_count[0]
        if idx < len(self._responses):
            response = self._responses[idx]
        else:
            response = AIMessage(content="Mock response (no more scripted replies).")
        self._shared_count[0] += 1
        return response


@pytest.fixture(autouse=True)
def _clear_llm_cache():
    """Clear the LLM cache before each test to avoid cross-test pollution."""
    from graphs import l1_reactor
    l1_reactor._llm_cache.clear()
    yield
    l1_reactor._llm_cache.clear()


@pytest.fixture
def mock_llm_simple():
    """A mock LLM that returns a simple text response (no tool calls)."""
    return MockLLM([AIMessage(content="Hello! I'm the mock LLM.")])


@pytest.fixture
def mock_llm_with_tool_call():
    """A mock LLM that first calls echo_tool, then returns a final answer."""
    return MockLLM([
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="echo_tool", args={"text": "test"}, id="call_1")
            ],
        ),
        AIMessage(content="The echo tool returned: Echo: test"),
    ])
