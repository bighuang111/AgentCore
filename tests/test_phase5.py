"""Phase 5 tests: plugin system — file tools, registry, MCP adapter, memory."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from plugins.tools.file_tools import (
    get_file_tools,
    inspect_dir,
    read_file,
    search_in_files,
    write_file,
)
from plugins.tools.registry import ToolRegistry, get_global_registry
from plugins.tools.mcp_adapter import MCPAdapter
from plugins.memory.sqlite_store import get_sqlite_checkpointer
from plugins.memory.user_profile import UserProfile


# --- File tools ---

class TestFileTools:
    def test_read_file(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello world")
        result = read_file.invoke({"file_path": str(p)})
        assert result == "hello world"

    def test_read_file_not_found(self):
        result = read_file.invoke({"file_path": "/nonexistent/file.txt"})
        assert "Error" in result

    def test_write_file(self, tmp_path):
        p = tmp_path / "output.txt"
        result = write_file.invoke({"file_path": str(p), "content": "new content"})
        assert "Successfully wrote" in result
        assert p.read_text() == "new content"

    def test_write_file_creates_dirs(self, tmp_path):
        p = tmp_path / "sub" / "dir" / "file.txt"
        write_file.invoke({"file_path": str(p), "content": "deep"})
        assert p.read_text() == "deep"

    def test_inspect_dir(self, tmp_path):
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "subdir").mkdir()
        result = inspect_dir.invoke({"dir_path": str(tmp_path)})
        assert "a.txt" in result
        assert "subdir" in result

    def test_search_in_files(self, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    return 'world'\n")
        (tmp_path / "b.py").write_text("x = 1\n")
        result = search_in_files.invoke({"dir_path": str(tmp_path), "pattern": "hello"})
        assert "hello" in result

    def test_search_no_match(self, tmp_path):
        (tmp_path / "a.txt").write_text("nothing here")
        result = search_in_files.invoke({"dir_path": str(tmp_path), "pattern": "zzzzz"})
        assert "No matches" in result

    def test_get_file_tools_count(self):
        tools = get_file_tools()
        assert len(tools) == 4


# --- Tool Registry ---

class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        reg.register_many(get_file_tools())
        assert reg.get("read_file") is not None
        assert reg.get("nonexistent") is None

    def test_list_names(self):
        reg = ToolRegistry()
        reg.register_many(get_file_tools())
        names = reg.list_names()
        assert "read_file" in names
        assert "write_file" in names

    def test_get_many(self):
        reg = ToolRegistry()
        reg.register_many(get_file_tools())
        tools = reg.get_many(["read_file", "write_file", "fake_tool"])
        assert len(tools) == 2

    def test_introductions(self):
        reg = ToolRegistry()
        reg.register_many(get_file_tools())
        intro = reg.introductions()
        assert "read_file" in intro
        assert "write_file" in intro

    def test_global_registry_loads_defaults(self):
        reg = get_global_registry()
        names = reg.list_names()
        assert "echo_tool" in names
        assert "read_file" in names

    def test_config_based_loading(self):
        """Factory should resolve tools from registry via config."""
        from core.config import AppConfig
        from core.factory import _resolve_tools

        config = AppConfig(enabled_tools=["echo_tool", "read_file"])
        tools = _resolve_tools(config)
        tool_names = [t.name for t in tools]
        assert "echo_tool" in tool_names
        assert "read_file" in tool_names


# --- MCP Adapter ---

class TestMCPAdapter:
    def test_empty_config(self):
        adapter = MCPAdapter()
        assert not adapter.is_loaded
        assert adapter.tools == []

    def test_load_empty_servers(self):
        adapter = MCPAdapter(server_configs=[])
        tools = adapter.load_tools()
        assert tools == []
        assert adapter.is_loaded


# --- Memory: SQLite ---

class TestSQLiteStore:
    def test_memory_saver_default(self):
        saver = get_sqlite_checkpointer()
        assert saver is not None

    def test_memory_saver_persistence(self):
        """MemorySaver should be usable as a checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver
        saver = get_sqlite_checkpointer()
        assert isinstance(saver, MemorySaver)


# --- Memory: UserProfile ---

class TestUserProfile:
    def test_set_and_get(self, tmp_path):
        path = tmp_path / "profile.json"
        profile = UserProfile(str(path))
        profile.set("theme", "dark")
        assert profile.get("theme") == "dark"

    def test_persistence(self, tmp_path):
        path = tmp_path / "profile.json"
        p1 = UserProfile(str(path))
        p1.set("lang", "en")

        p2 = UserProfile(str(path))
        assert p2.get("lang") == "en"

    def test_delete(self, tmp_path):
        path = tmp_path / "profile.json"
        profile = UserProfile(str(path))
        profile.set("key", "val")
        profile.delete("key")
        assert profile.get("key") is None

    def test_all(self, tmp_path):
        path = tmp_path / "profile.json"
        profile = UserProfile(str(path))
        profile.set("a", 1)
        profile.set("b", 2)
        assert profile.all() == {"a": 1, "b": 2}
