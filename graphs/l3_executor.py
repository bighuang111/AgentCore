"""L3 Autonomous Executor: Closed-loop with plan/execute/evaluate/reflect/revise.

Uses Command(goto=...) for dynamic routing from evaluate node,
and RetryPolicy for fault tolerance on all nodes.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from core.state import L3State

DEFAULT_RETRY = RetryPolicy(max_attempts=3)


def _default_plan(state: L3State) -> dict:
    return {
        "messages": [AIMessage(content="[L3] Planning phase.")],
        "current_tier": "l3",
    }


def _default_execute(state: L3State) -> dict:
    return {
        "messages": [AIMessage(content="[L3] Executing plan.")],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def _default_reflect(state: L3State) -> dict:
    log = list(state.get("reflection_log", []))
    iteration = state.get("iteration_count", 0)
    log.append(f"Reflection at iteration {iteration}: quality={state.get('quality_score', 0)}")
    return {
        "messages": [AIMessage(content=f"[L3] Reflecting. Score: {state.get('quality_score', 0)}")],
        "reflection_log": log,
    }


def _default_revise(state: L3State) -> dict:
    return {
        "messages": [AIMessage(content="[L3] Revising approach based on reflection.")],
    }


def _make_evaluate_with_command(
    quality_threshold: float = 0.8,
    max_iterations: int = 5,
    evaluate_fn: Callable | None = None,
) -> Callable:
    """Create an evaluate node that uses Command(goto=...) for dynamic routing.

    This merges evaluation logic and routing into a single node,
    replacing the previous add_conditional_edges approach.
    """

    def evaluate_and_route(state: L3State) -> Command[Literal["reflect", "__end__"]]:
        # Run the actual evaluation logic
        if evaluate_fn is not None:
            eval_result = evaluate_fn(state)
        else:
            eval_result = {
                "messages": [AIMessage(content="[L3] Evaluating results.")],
                "quality_score": state.get("quality_score", 0.0),
            }

        # Determine quality_score from the eval result
        quality = eval_result.get("quality_score", state.get("quality_score", 0.0))
        iteration = state.get("iteration_count", 0)
        is_complete = state.get("is_complete", False)

        # Route decision
        if is_complete or quality >= quality_threshold or iteration >= max_iterations:
            return Command(update=eval_result, goto=END)
        else:
            return Command(update=eval_result, goto="reflect")

    return evaluate_and_route


def build_l3_graph(
    quality_threshold: float = 0.8,
    max_iterations: int = 5,
    plan_fn: Callable | None = None,
    execute_fn: Callable | None = None,
    evaluate_fn: Callable | None = None,
    reflect_fn: Callable | None = None,
    revise_fn: Callable | None = None,
):
    """Build the L3 autonomous executor graph.

    Flow: plan → execute → evaluate →(Command: quality ok?)→ END
                  ↑                  →(Command: not ok?)→ reflect → revise ─┘

    Uses Command(goto=...) for dynamic routing (no conditional_edges needed).
    Uses RetryPolicy for fault tolerance on all nodes.
    """
    evaluate_node = _make_evaluate_with_command(
        quality_threshold, max_iterations, evaluate_fn
    )

    graph = StateGraph(L3State)

    graph.add_node("plan", plan_fn or _default_plan, retry_policy=DEFAULT_RETRY)
    graph.add_node("execute", execute_fn or _default_execute, retry_policy=DEFAULT_RETRY)
    graph.add_node("evaluate", evaluate_node, retry_policy=DEFAULT_RETRY)
    graph.add_node("reflect", reflect_fn or _default_reflect, retry_policy=DEFAULT_RETRY)
    graph.add_node("revise", revise_fn or _default_revise, retry_policy=DEFAULT_RETRY)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "evaluate")
    # evaluate uses Command(goto=...) — no conditional_edges needed
    graph.add_edge("reflect", "revise")
    graph.add_edge("revise", "execute")

    return graph.compile()
