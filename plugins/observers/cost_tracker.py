"""Token consumption and cost tracking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UsageRecord:
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CostTracker:
    """Track token usage across LLM calls."""

    # Approximate cost per 1K tokens (USD)
    DEFAULT_PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }

    def __init__(self):
        self._records: list[UsageRecord] = []

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self._records.append(UsageRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ))

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self._records)

    @property
    def total_calls(self) -> int:
        return len(self._records)

    def estimated_cost(self) -> float:
        """Estimate total cost in USD based on default pricing."""
        total = 0.0
        for r in self._records:
            pricing = self.DEFAULT_PRICING.get(r.model, {"input": 0.001, "output": 0.002})
            total += (r.prompt_tokens / 1000) * pricing["input"]
            total += (r.completion_tokens / 1000) * pricing["output"]
        return round(total, 6)

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost(),
            "records": [
                {"model": r.model, "prompt": r.prompt_tokens, "completion": r.completion_tokens}
                for r in self._records
            ],
        }
