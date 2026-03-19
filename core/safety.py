"""Safety guardrails: budget control, content filtering, PII sanitization."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from core.state import BaseState


class BudgetGuard:
    """Loop-count budget guard to prevent infinite ReAct cycles."""

    def __init__(self, max_loops: int = 10):
        self.max_loops = max_loops

    def should_continue(self, state: BaseState) -> str:
        """Check budget, then decide whether to continue to tools or end.

        Returns:
            "tool_node" if tool calls exist and budget allows.
            "__end__" otherwise.
        """
        llm_call_count = state.get("llm_call_count", 0)
        if llm_call_count >= self.max_loops:
            return "__end__"

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tool_node"
        return "__end__"


class PIISanitizer:
    """Bidirectional PII filter: mask sensitive data in text."""

    # Patterns for common PII types
    PATTERNS = {
        "email": (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
        "phone": (re.compile(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE]"),
        "ssn": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
        "credit_card": (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CREDIT_CARD]"),
        "ip_address": (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "[IP_ADDRESS]"),
    }

    def __init__(self, enabled_patterns: list[str] | None = None):
        if enabled_patterns is None:
            self._active = dict(self.PATTERNS)
        else:
            self._active = {k: v for k, v in self.PATTERNS.items() if k in enabled_patterns}

    def sanitize(self, text: str) -> str:
        """Replace PII patterns with placeholders."""
        for _, (pattern, replacement) in self._active.items():
            text = pattern.sub(replacement, text)
        return text
