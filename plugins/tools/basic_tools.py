"""Basic demonstration tools: echo and current_time."""

from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import tool


@tool
def echo_tool(text: str) -> str:
    """Echo back the input text. Useful for testing."""
    return f"Echo: {text}"


@tool
def current_time() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def get_basic_tools() -> list:
    """Return the list of basic built-in tools."""
    return [echo_tool, current_time]
