"""L1 Atomic Reactor: Single-hop ReAct pattern for immediate Q&A tasks.

Features:
- Multi-provider LLM support via init_chat_model (OpenAI/Gemini/Claude/DeepSeek).
- RetryPolicy on LLM and tool nodes for fault tolerance.
- context_schema for runtime configuration injection.
- Runtime model switching via context.llm_model.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy

from core.runtime_config import RuntimeContext
from core.safety import BudgetGuard
from core.state import BaseState
from plugins.tools.basic_tools import get_basic_tools

DEFAULT_RETRY = RetryPolicy(max_attempts=3)

# Module-level LLM cache (lazy-initialized)
_llm_cache: dict[str, BaseChatModel] = {}


def _get_llm(model: str, temperature: float = 0.7) -> BaseChatModel:
    """Get or create a cached LLM instance."""
    cache_key = f"{model}:{temperature}"
    if cache_key not in _llm_cache:
        from core.llm import create_llm
        _llm_cache[cache_key] = create_llm(model, temperature=temperature)
    return _llm_cache[cache_key]


def _make_llm_call(default_model: str, tools: list):
    """Create an llm_call node function bound to a default model and tools."""

    def llm_call(state: BaseState, *, runtime=None) -> dict:
        """Invoke the LLM with tools. Supports runtime model override."""
        ctx = runtime.context if runtime else {}
        model = ctx.get("llm_model", default_model) if ctx else default_model
        temperature = ctx.get("temperature", 0.7) if ctx else 0.7

        try:
            llm = _get_llm(model, temperature)
            llm_with_tools = llm.bind_tools(tools)
            response = llm_with_tools.invoke(state["messages"])
        except Exception as e:
            response = AIMessage(content=f"LLM error: {e}")

        return {
            "messages": [response],
            "llm_call_count": state.get("llm_call_count", 0) + 1,
        }

    return llm_call


def build_l1_graph(
    max_loops: int = 10,
    tools: list | None = None,
    model: str = "openai:gpt-4o-mini",
):
    """Build the L1 ReAct graph with multi-provider LLM support.

    Args:
        max_loops: Maximum LLM call iterations before forced stop.
        tools: Optional tool list. Defaults to basic tools.
        model: Default LLM model string (e.g. "openai:gpt-4o-mini",
               "google_genai:gemini-2.5-flash", "anthropic:claude-sonnet-4-6").
               Can be overridden at runtime via context.llm_model.
    """
    tool_list = tools or get_basic_tools()
    tool_node = ToolNode(tool_list)
    guard = BudgetGuard(max_loops=max_loops)
    llm_call = _make_llm_call(model, tool_list)

    graph = StateGraph(BaseState, context_schema=RuntimeContext)
    graph.add_node("llm_call", llm_call, retry_policy=DEFAULT_RETRY)
    graph.add_node("tool_node", tool_node, retry_policy=DEFAULT_RETRY)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", guard.should_continue)
    graph.add_edge("tool_node", "llm_call")

    return graph.compile()
