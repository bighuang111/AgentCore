"""Application configuration: parse user_config.yaml into Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SafetyConfig(BaseModel):
    max_loops: int = Field(default=10, ge=1)
    max_tokens: int = Field(default=4096, ge=1)


class AppConfig(BaseModel):
    default_llm: str = "openai:gpt-4o-mini"
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    enabled_tiers: list[str] = Field(default_factory=lambda: ["l1"])
    enabled_tools: list[str] = Field(default_factory=lambda: ["echo_tool", "current_time"])

    @classmethod
    def from_yaml(cls, path: str | Path = "user_config.yaml") -> AppConfig:
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
