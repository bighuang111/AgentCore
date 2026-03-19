"""Phase 6 tests: monitoring, StandardOutput, PII sanitization."""

from __future__ import annotations

import json
import os

from langchain_core.messages import AIMessage, HumanMessage

from core.output import StandardOutput
from core.safety import PIISanitizer
from plugins.observers.langsmith_tracer import configure_langsmith, is_tracing_enabled
from plugins.observers.cost_tracker import CostTracker
from plugins.observers.logger import JSONFormatter, get_logger


# --- StandardOutput ---

class TestStandardOutput:
    def _make_state(self, content: str = "Hello world") -> dict:
        return {
            "messages": [
                HumanMessage(content="Hi"),
                AIMessage(content=content),
            ],
            "metadata": {"source": "test"},
            "llm_call_count": 1,
            "current_tier": "l1",
        }

    def test_to_markdown(self):
        state = self._make_state("Final answer here.")
        md = StandardOutput.to_markdown(state)
        assert "Final answer here." in md
        assert "Tier: l1" in md
        assert "LLM calls: 1" in md

    def test_to_json(self):
        state = self._make_state("JSON output")
        result = StandardOutput.to_json(state)
        assert result["tier"] == "l1"
        assert result["output"] == "JSON output"
        assert result["llm_call_count"] == 1

    def test_to_json_string(self):
        state = self._make_state("Stringified")
        s = StandardOutput.to_json_string(state)
        parsed = json.loads(s)
        assert parsed["output"] == "Stringified"

    def test_empty_messages(self):
        state = {"messages": [], "metadata": {}, "llm_call_count": 0, "current_tier": "l1"}
        md = StandardOutput.to_markdown(state)
        assert "No output" in md


# --- PII Sanitizer ---

class TestPIISanitizer:
    def test_email(self):
        s = PIISanitizer()
        assert s.sanitize("Contact me at john@example.com") == "Contact me at [EMAIL]"

    def test_phone(self):
        s = PIISanitizer()
        assert "[PHONE]" in s.sanitize("Call 123-456-7890")

    def test_ssn(self):
        s = PIISanitizer()
        assert s.sanitize("SSN: 123-45-6789") == "SSN: [SSN]"

    def test_credit_card(self):
        s = PIISanitizer()
        result = s.sanitize("Card: 4111-1111-1111-1111")
        assert "[CREDIT_CARD]" in result

    def test_multiple_pii(self):
        s = PIISanitizer()
        text = "Email: a@b.com, Phone: 555-123-4567, SSN: 111-22-3333"
        result = s.sanitize(text)
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "[SSN]" in result

    def test_selective_patterns(self):
        s = PIISanitizer(enabled_patterns=["email"])
        result = s.sanitize("a@b.com and 123-45-6789")
        assert "[EMAIL]" in result
        assert "123-45-6789" in result  # SSN not filtered

    def test_no_pii(self):
        s = PIISanitizer()
        text = "This is clean text with no PII."
        assert s.sanitize(text) == text


# --- LangSmith Tracer ---

class TestLangSmithTracer:
    def test_configure_disabled(self):
        result = configure_langsmith(enabled=False)
        assert result is False
        assert not is_tracing_enabled()

    def test_configure_enabled(self):
        result = configure_langsmith(project="test-proj", enabled=True)
        assert result is True
        assert is_tracing_enabled()
        assert os.environ.get("LANGCHAIN_PROJECT") == "test-proj"
        # Clean up
        configure_langsmith(enabled=False)


# --- Cost Tracker ---

class TestCostTracker:
    def test_record_and_summary(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 100, 50)
        tracker.record("gpt-4o-mini", 200, 100)
        assert tracker.total_calls == 2
        assert tracker.total_tokens == 450

    def test_estimated_cost(self):
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 1000, 1000)
        cost = tracker.estimated_cost()
        assert cost > 0

    def test_summary_structure(self):
        tracker = CostTracker()
        tracker.record("gpt-4o", 500, 200)
        summary = tracker.summary()
        assert "total_calls" in summary
        assert "estimated_cost_usd" in summary
        assert len(summary["records"]) == 1


# --- Structured Logger ---

class TestStructuredLogger:
    def test_json_format(self):
        formatter = JSONFormatter()
        import logging
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert "timestamp" in parsed

    def test_get_logger(self):
        logger = get_logger("test-logger")
        assert logger.name == "test-logger"
