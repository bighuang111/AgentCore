"""Tool registry: auto-discovery, self-introduction, config-based loading."""

from __future__ import annotations

from typing import Any, Callable


class ToolRegistry:
    """Central registry for all available tools with self-introduction."""

    def __init__(self):
        self._tools: dict[str, Any] = {}

    def register(self, tool_obj: Any) -> None:
        """Register a tool. Uses the tool's name attribute."""
        name = getattr(tool_obj, "name", None) or tool_obj.__name__
        self._tools[name] = tool_obj

    def register_many(self, tools: list) -> None:
        """Register multiple tools at once."""
        for t in tools:
            self.register(t)

    def get(self, name: str) -> Any | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_many(self, names: list[str]) -> list:
        """Get multiple tools by name, skipping unknown ones."""
        return [self._tools[n] for n in names if n in self._tools]

    def all_tools(self) -> list:
        """Return all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def introductions(self) -> str:
        """Generate self-introduction text for all registered tools."""
        lines = []
        for name, tool_obj in self._tools.items():
            doc = getattr(tool_obj, "description", None) or getattr(tool_obj, "__doc__", "No description.")
            lines.append(f"- **{name}**: {doc}")
        return "\n".join(lines)


# Global registry instance
_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry, populating it on first access."""
    if not _global_registry.list_names():
        _populate_defaults()
    return _global_registry


def _populate_defaults():
    """Populate the global registry with all built-in tools."""
    from plugins.tools.basic_tools import get_basic_tools
    from plugins.tools.file_tools import get_file_tools

    _global_registry.register_many(get_basic_tools())
    _global_registry.register_many(get_file_tools())
