"""File operation tools: read, write, inspect, search."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> str:
    """Read and return the contents of a file."""
    p = Path(file_path)
    if not p.exists():
        return f"Error: File not found: {file_path}"
    if not p.is_file():
        return f"Error: Not a file: {file_path}"
    return p.read_text(encoding="utf-8", errors="replace")


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {file_path}"


@tool
def inspect_dir(dir_path: str) -> str:
    """List the contents of a directory with file types and sizes."""
    p = Path(dir_path)
    if not p.exists():
        return f"Error: Directory not found: {dir_path}"
    if not p.is_dir():
        return f"Error: Not a directory: {dir_path}"

    entries = []
    for item in sorted(p.iterdir()):
        kind = "DIR" if item.is_dir() else "FILE"
        size = item.stat().st_size if item.is_file() else 0
        entries.append(f"  [{kind}] {item.name} ({size} bytes)")
    return f"Contents of {dir_path}:\n" + "\n".join(entries) if entries else f"{dir_path} is empty."


@tool
def search_in_files(dir_path: str, pattern: str) -> str:
    """Search for a text pattern in all files within a directory (recursive)."""
    p = Path(dir_path)
    if not p.exists():
        return f"Error: Directory not found: {dir_path}"

    results = []
    for fp in p.rglob("*"):
        if not fp.is_file():
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    results.append(f"{fp}:{i}: {line.strip()}")
        except Exception:
            continue

    if not results:
        return f"No matches for '{pattern}' in {dir_path}"
    return "\n".join(results[:50])  # Cap at 50 results


def get_file_tools() -> list:
    """Return the list of file operation tools."""
    return [read_file, write_file, inspect_dir, search_in_files]
