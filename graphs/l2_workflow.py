"""L2 Standard Workflow: Linear DAG with SOP steps and approval gates.

Uses interrupt() for human-in-the-loop approval (replacing interrupt_before).
Uses RetryPolicy for fault tolerance on action nodes.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from core.state import L2State

DEFAULT_RETRY = RetryPolicy(max_attempts=3)


def _make_step_node(step: dict):
    """Create a node function for a workflow step.

    For approval-type steps, uses interrupt() to pause and collect
    the approver's response, which is stored in approval_status.
    """
    step_name = step["name"]
    step_type = step.get("type", "action")

    if step_type == "approval":
        def node_fn(state: L2State) -> dict:
            # Pause and wait for human approval; the resume value is returned
            approval = interrupt(f"Approval required for step: {step_name}")
            return {
                "current_step": step_name,
                "messages": [AIMessage(content=f"[L2] Approval step '{step_name}': {approval}")],
                "approval_status": str(approval),
                "artifacts": {**state.get("artifacts", {}), step_name: "approved"},
            }
    else:
        def node_fn(state: L2State) -> dict:
            return {
                "current_step": step_name,
                "messages": [AIMessage(content=f"[L2] Executing step: {step_name}")],
                "artifacts": {**state.get("artifacts", {}), step_name: "completed"},
            }

    node_fn.__name__ = step_name
    return node_fn


def build_l2_graph(steps: list[dict[str, Any]], checkpointer=None):
    """Build a linear DAG workflow from a list of SOP steps.

    Args:
        steps: List of step definitions, each with:
            - name: Step identifier
            - type: "action" (default) or "approval" (triggers interrupt())
        checkpointer: Optional checkpointer for state persistence.
            Required for interrupt() to work. If None, compiled without checkpointer.

    Example:
        steps = [
            {"name": "gather_info", "type": "action"},
            {"name": "review", "type": "approval"},
            {"name": "generate_report", "type": "action"},
        ]
    """
    if not steps:
        raise ValueError("L2 workflow requires at least one step.")

    graph = StateGraph(L2State)

    for step in steps:
        node_fn = _make_step_node(step)
        policy = DEFAULT_RETRY if step.get("type") != "approval" else None
        graph.add_node(step["name"], node_fn, retry_policy=policy)

    # Wire linear edges: START -> step1 -> step2 -> ... -> END
    graph.add_edge(START, steps[0]["name"])
    for i in range(len(steps) - 1):
        graph.add_edge(steps[i]["name"], steps[i + 1]["name"])
    graph.add_edge(steps[-1]["name"], END)

    return graph.compile(checkpointer=checkpointer)
