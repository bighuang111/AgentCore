"""LangSmith tracer integration for full-chain observability."""

from __future__ import annotations

import os


def configure_langsmith(
    project: str = "omni-harness",
    enabled: bool | None = None,
) -> bool:
    """Configure LangSmith tracing via environment variables.

    Args:
        project: LangSmith project name.
        enabled: Force enable/disable. If None, reads from env.

    Returns:
        True if tracing is enabled.
    """
    if enabled is None:
        enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    if enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", project)
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    return enabled


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is currently enabled."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
