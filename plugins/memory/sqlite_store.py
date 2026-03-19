"""SQLite-based checkpointer for short-term memory (session context)."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver


def get_sqlite_checkpointer(db_path: str | None = None):
    """Get a checkpointer for state persistence.

    Args:
        db_path: Path to SQLite database. If None, uses in-memory storage.

    Returns:
        A LangGraph-compatible checkpointer.
    """
    if db_path is None:
        return MemorySaver()

    # For SQLite file-based persistence, use langgraph's sqlite saver
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver.from_conn_string(db_path)
    except ImportError:
        # Fallback to memory saver if sqlite extra not installed
        return MemorySaver()
