"""L1 Atomic Reactor: Single-hop ReAct pattern for immediate Q&A tasks.

Features:
- RetryPolicy on LLM and tool nodes for fault tolerance.
- context_schema for runtime configuration injection.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from core.runtime_config import RuntimeContext
from core.safety import BudgetGuard
from core.state import BaseState
from plugins.tools.basic_tools import get_basic_tools

DEFAULT_RETRY = RetryPolicy(max_attempts=3)


def _llm_call(state: BaseState, *, runtime=None) -> dict:
    """Invoke the LLM with the current messages and tools.

    Accepts optional runtime context for per-invocation configuration.
    """
    llm = state.get("_llm")
    tools = get_basic_tools()

    # Runtime context can override behavior
    ctx = runtime.context if runtime else {}
    model_override = ctx.get("llm_model") if ctx else None

    if llm is not None:
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(state["messages"])
    else:
        response = AIMessage(content="No LLM configured.")

    return {
        "messages": [response],
        "llm_call_count": state.get("llm_call_count", 0) + 1,
    }


def build_l1_graph(max_loops: int = 10, tools: list | None = None):
    """Build the L1 ReAct graph with budget control, retry, and runtime config.

    Args:
        max_loops: Maximum LLM call iterations before forced stop.
        tools: Optional tool list. Defaults to basic tools.
    """
    tool_list = tools or get_basic_tools()
    tool_node = ToolNode(tool_list)
    guard = BudgetGuard(max_loops=max_loops)

    graph = StateGraph(BaseState, context_schema=RuntimeContext)
    graph.add_node("llm_call", _llm_call, retry_policy=DEFAULT_RETRY)
    graph.add_node("tool_node", tool_node, retry_policy=DEFAULT_RETRY)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", guard.should_continue)
    graph.add_edge("tool_node", "llm_call")

    return graph.compile()
