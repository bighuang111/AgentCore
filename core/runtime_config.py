"""Runtime configuration: context_schema for injecting per-invocation parameters."""

from __future__ import annotations

from typing_extensions import TypedDict


class RuntimeContext(TypedDict, total=False):
    """Context schema injected at graph invocation time.

    Nodes access this via `runtime.context` parameter.
    Fields are optional (total=False) so callers only provide what they need.
    """

    user_id: str
    llm_model: str
    temperature: float
    language: str
    session_id: str
