"""User profile: cross-session preference storage (JSON file)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class UserProfile:
    """Persistent user preferences stored as a JSON file."""

    def __init__(self, path: str = "user_profile.json"):
        self._path = Path(path)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save()

    def delete(self, key: str) -> None:
        self._data.pop(key, None)
        self._save()

    def all(self) -> dict[str, Any]:
        return dict(self._data)
