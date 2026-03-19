"""Universal LLM initialization: multi-provider support via init_chat_model.

Supported model string formats:
    "gpt-4o"                        → OpenAI (auto-detected)
    "openai:gpt-4o-mini"            → OpenAI (explicit)
    "google_genai:gemini-2.5-flash" → Google Gemini
    "anthropic:claude-sonnet-4-6"   → Anthropic Claude
    "deepseek:deepseek-chat"        → DeepSeek

Environment variables (set in .env):
    OPENAI_API_KEY       → OpenAI
    GOOGLE_API_KEY       → Google Gemini
    ANTHROPIC_API_KEY    → Anthropic
    DEEPSEEK_API_KEY     → DeepSeek (via OpenAI-compatible endpoint)

Usage:
    from core.llm import create_llm, create_configurable_llm

    # Static model
    llm = create_llm("openai:gpt-4o-mini")

    # Configurable model (switch at runtime)
    llm = create_configurable_llm(temperature=0)
    llm.invoke("Hi", config={"configurable": {"model": "google_genai:gemini-2.5-flash"}})
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel


def create_llm(
    model: str = "openai:gpt-4o-mini",
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a chat model instance from a provider:model string.

    Args:
        model: Model identifier. Formats:
            - "gpt-4o"                         (auto-detect provider)
            - "openai:gpt-4o-mini"             (explicit provider)
            - "google_genai:gemini-2.5-flash"  (Google Gemini)
            - "anthropic:claude-sonnet-4-6"    (Anthropic)
            - "deepseek:deepseek-chat"         (DeepSeek)
        temperature: Sampling temperature.
        **kwargs: Additional model parameters.

    Returns:
        A LangChain BaseChatModel instance.
    """
    # Handle DeepSeek via OpenAI-compatible endpoint
    if model.startswith("deepseek:"):
        model_name = model.split(":", 1)[1]
        return _create_deepseek(model_name, temperature, **kwargs)

    from langchain.chat_models import init_chat_model
    return init_chat_model(model, temperature=temperature, **kwargs)


def create_configurable_llm(
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a configurable model that can switch providers at runtime.

    Usage:
        llm = create_configurable_llm(temperature=0)
        # Use with different models at invocation time:
        llm.invoke("Hi", config={"configurable": {"model": "openai:gpt-4o"}})
        llm.invoke("Hi", config={"configurable": {"model": "anthropic:claude-sonnet-4-6"}})
    """
    from langchain.chat_models import init_chat_model
    return init_chat_model(temperature=temperature, configurable_fields="any", **kwargs)


def _create_deepseek(
    model_name: str,
    temperature: float,
    **kwargs: Any,
) -> BaseChatModel:
    """Create DeepSeek model via OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        temperature=temperature,
        **kwargs,
    )


def list_available_providers() -> dict[str, bool]:
    """Check which providers have API keys configured."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "google_genai": bool(os.environ.get("GOOGLE_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "deepseek": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }
