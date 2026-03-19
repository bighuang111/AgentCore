"""StandardOutput: format agent results as Markdown or JSON."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from core.state import BaseState


def _extract_content(msg: AIMessage) -> str:
    """Extract text content from an AIMessage, handling list-type content (Gemini)."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


class StandardOutput:
    """Standardize agent output into consumable formats."""

    @staticmethod
    def to_markdown(state: BaseState) -> str:
        """Extract the final AI message and format as Markdown."""
        messages = state.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
        if not ai_messages:
            return "_No output generated._"

        last_content = _extract_content(ai_messages[-1])
        lines = [
            f"## Agent Output (Tier: {state.get('current_tier', 'unknown')})",
            "",
            last_content,
            "",
            f"---",
            f"_LLM calls: {state.get('llm_call_count', 0)}_",
        ]
        return "\n".join(lines)

    @staticmethod
    def to_json(state: BaseState) -> dict[str, Any]:
        """Extract structured output from the final state."""
        messages = state.get("messages", [])
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]

        output = _extract_content(ai_messages[-1]) if ai_messages else ""
        return {
            "tier": state.get("current_tier", "unknown"),
            "output": output,
            "llm_call_count": state.get("llm_call_count", 0),
            "metadata": state.get("metadata", {}),
        }

    @staticmethod
    def to_json_string(state: BaseState) -> str:
        """Return JSON-formatted string."""
        return json.dumps(StandardOutput.to_json(state), indent=2, ensure_ascii=False)
