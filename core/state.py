"""Global unified state definitions."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class BaseState(TypedDict):
    """Base state shared across all orchestration tiers."""

    messages: Annotated[list[AnyMessage], add_messages]
    metadata: dict[str, Any]
    llm_call_count: int
    current_tier: str


class L2State(BaseState):
    """Extended state for L2 workflow with SOP steps."""

    current_step: str
    approval_status: str
    artifacts: dict[str, Any]


class L3State(BaseState):
    """Extended state for L3 autonomous executor with reflection."""

    iteration_count: int
    reflection_log: list[str]
    quality_score: float
    is_complete: bool
