"""L1 Reactor Demo: Multi-provider LLM with tool calling.

Usage:
    # 1. Fill in your API keys in .env
    # 2. Run one of the demos:

    python examples/l1_demo.py explorer    # Code explorer (inspect_dir + read_file)
    python examples/l1_demo.py processor   # File processor (read YAML → write JSON)
    python examples/l1_demo.py analyzer    # Code analyzer (search_in_files)
    python examples/l1_demo.py providers   # Check which providers are configured

    # Override model at runtime:
    MODEL=google_genai:gemini-2.5-flash python examples/l1_demo.py explorer
    MODEL=anthropic:claude-sonnet-4-6 python examples/l1_demo.py processor
    MODEL=deepseek:deepseek-chat python examples/l1_demo.py analyzer
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage
from core.config import AppConfig, SafetyConfig
from core.factory import build_graph
from core.llm import list_available_providers
from core.output import StandardOutput


def _get_model() -> str:
    """Get model from env or default."""
    return os.environ.get("MODEL", "openai:gpt-4o-mini")


def _print_result(result: dict) -> None:
    """Pretty-print the result."""
    print("\n" + "=" * 60)
    print(StandardOutput.to_markdown(result))
    print("=" * 60)
    print(f"\nLLM calls: {result['llm_call_count']}")
    print(f"Messages in history: {len(result['messages'])}")


# ============================================================
# Demo 1: Code Explorer
# ============================================================

def demo_explorer():
    """Explore project structure using inspect_dir + read_file."""
    print(f"[Code Explorer] Model: {_get_model()}")
    print("-" * 40)

    config = AppConfig(
        safety=SafetyConfig(max_loops=8),
        enabled_tools=["inspect_dir", "read_file", "search_in_files"],
    )
    graph = build_graph("l1", config=config, model=_get_model())

    result = graph.invoke({
        "messages": [HumanMessage(content=(
            f"查看 {PROJECT_ROOT}/core/ 目录下有哪些文件，"
            "然后读取 state.py，告诉我 L3State 有哪些字段，各自的作用是什么。"
        ))],
        "metadata": {"demo": "explorer"},
        "llm_call_count": 0,
        "current_tier": "l1",
    })
    _print_result(result)


# ============================================================
# Demo 2: File Processor (YAML → JSON)
# ============================================================

def demo_processor():
    """Read YAML config and convert to JSON."""
    print(f"[File Processor] Model: {_get_model()}")
    print("-" * 40)

    config = AppConfig(
        safety=SafetyConfig(max_loops=6),
        enabled_tools=["read_file", "write_file"],
    )
    graph = build_graph("l1", config=config, model=_get_model())

    result = graph.invoke({
        "messages": [HumanMessage(content=(
            f"读取 {PROJECT_ROOT}/user_config.yaml 的内容，"
            "把它转换成等价的 JSON 格式，写入到 /tmp/user_config.json 文件中。"
            "完成后告诉我写入了什么内容。"
        ))],
        "metadata": {"demo": "processor"},
        "llm_call_count": 0,
        "current_tier": "l1",
    })
    _print_result(result)

    # Verify output
    output_path = Path("/tmp/user_config.json")
    if output_path.exists():
        print(f"\n[Verification] /tmp/user_config.json content:")
        print(output_path.read_text())


# ============================================================
# Demo 3: Code Analyzer
# ============================================================

def demo_analyzer():
    """Search for patterns in code and generate a report."""
    print(f"[Code Analyzer] Model: {_get_model()}")
    print("-" * 40)

    config = AppConfig(
        safety=SafetyConfig(max_loops=10),
        enabled_tools=["search_in_files", "read_file", "inspect_dir"],
    )
    graph = build_graph("l1", config=config, model=_get_model())

    result = graph.invoke({
        "messages": [HumanMessage(content=(
            f"在 {PROJECT_ROOT}/graphs/ 目录中，"
            "搜索所有使用了 'RetryPolicy' 的地方，然后搜索所有使用了 'Command' 的地方。"
            "给我一份简要的中文报告：哪些文件用了什么高级特性，各自用来做什么。"
        ))],
        "metadata": {"demo": "analyzer"},
        "llm_call_count": 0,
        "current_tier": "l1",
    })
    _print_result(result)


# ============================================================
# Provider Check
# ============================================================

def show_providers():
    """Show which LLM providers have API keys configured."""
    providers = list_available_providers()
    print("Configured LLM Providers:")
    print("-" * 40)
    for name, available in providers.items():
        status = "ready" if available else "not configured"
        icon = "[OK]" if available else "[--]"
        print(f"  {icon} {name:15s} {status}")

    configured = [k for k, v in providers.items() if v]
    if not configured:
        print("\nNo providers configured! Fill in API keys in .env file.")
    else:
        print(f"\nYou can use: MODEL={configured[0]}:model-name python examples/l1_demo.py ...")


# ============================================================
# Main
# ============================================================

DEMOS = {
    "explorer": demo_explorer,
    "processor": demo_processor,
    "analyzer": demo_analyzer,
    "providers": show_providers,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in DEMOS:
        print("Usage: python examples/l1_demo.py <demo>")
        print(f"Available demos: {', '.join(DEMOS.keys())}")
        print("\nExamples:")
        print("  python examples/l1_demo.py providers")
        print("  python examples/l1_demo.py explorer")
        print("  MODEL=google_genai:gemini-2.5-flash python examples/l1_demo.py explorer")
        sys.exit(1)

    DEMOS[sys.argv[1]]()
