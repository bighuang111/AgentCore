"""Shared test fixtures with Mock LLM support."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall


class MockLLM:
    """A mock LLM that returns predefined responses.

    Supports tool binding and can simulate tool calls.
    """

    def __init__(self, responses: list[AIMessage] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0
        self._bound_tools: list = []

    def bind_tools(self, tools: list) -> "MockLLM":
        clone = MockLLM(self._responses)
        clone._bound_tools = tools
        clone._call_count = self._call_count
        return clone

    def invoke(self, messages: Any, **kwargs) -> AIMessage:
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = AIMessage(content="Mock response (no more scripted replies).")
        self._call_count += 1
        return response


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
