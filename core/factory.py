"""Graph factory: build L1/L2/L3 graph instances from configuration."""

from __future__ import annotations

from core.config import AppConfig
from graphs.l1_reactor import build_l1_graph
from graphs.l2_workflow import build_l2_graph
from graphs.l3_executor import build_l3_graph
from plugins.tools.basic_tools import get_basic_tools
from plugins.tools.registry import get_global_registry


def _resolve_tools(config: AppConfig) -> list:
    """Resolve tool instances from config's enabled_tools via the registry."""
    registry = get_global_registry()
    tools = registry.get_many(config.enabled_tools)
    return tools or get_basic_tools()


def build_graph(tier: str = "l1", config: AppConfig | None = None, **kwargs):
    """Build and return a compiled graph for the given tier."""
    if config is None:
        config = AppConfig.from_yaml()

    tools = _resolve_tools(config)

    builders = {
        "l1": lambda: build_l1_graph(
            max_loops=config.safety.max_loops,
            tools=tools,
            model=kwargs.get("model", config.default_llm),
        ),
        "l2": lambda: build_l2_graph(steps=kwargs.get("steps", [])),
        "l3": lambda: build_l3_graph(
            quality_threshold=kwargs.get("quality_threshold", 0.8),
            max_iterations=kwargs.get("max_iterations", 5),
            **{k: v for k, v in kwargs.items() if k.endswith("_fn")},
        ),
    }
    builder = builders.get(tier)
    if builder is None:
        raise ValueError(f"Unknown tier: {tier}. Available: {list(builders.keys())}")
    return builder()


# Module-level variables for langgraph CLI discovery
l1_graph = build_l1_graph()

_default_l2_steps = [
    {"name": "step_input", "type": "action"},
    {"name": "step_process", "type": "action"},
    {"name": "step_output", "type": "action"},
]
l2_graph = build_l2_graph(steps=_default_l2_steps)

l3_graph = build_l3_graph()
