---
name: agent-builder
description: Build custom AI Agents on the Omni-Harness framework. Use when user asks to create, build, develop, or implement any Agent, bot, assistant, or automated workflow. Covers tool creation, graph orchestration (L1/L2/L3), config registration, and testing.
argument-hint: [agent description in natural language]
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, Agent
---

# Omni-Harness Agent Builder

You are building an Agent on top of the **Omni-Harness** framework. This project provides three orchestration tiers, a pluggable tool system, safety guardrails, and multi-provider LLM support.

**Before writing any code**, read this skill completely. Then follow the workflow step by step.

## Step 0: Understand the Request

Parse `$ARGUMENTS` (the user's natural language description) and determine:

1. **What the Agent should do** — the core task
2. **What external services it needs** — APIs, email, databases, web scraping, etc.
3. **Whether human approval is needed** — sending emails, publishing, payments, etc.
4. **Whether iteration/quality is needed** — research, writing, optimization loops

## Step 1: Choose the Orchestration Tier

Use this decision tree:

```
Single-step Q&A or tool call?
  → YES: L1 (Atomic Reactor)

Multi-step with fixed SOP?
  → YES: L2 (Standard Workflow)
  → Has approval gates? → Add type="approval" steps

Needs self-reflection / quality iteration?
  → YES: L3 (Autonomous Executor)
```

| Tier | When to Use | Examples |
|------|-------------|---------|
| **L1** | One question, one answer, maybe with tools | Translation, file conversion, single query |
| **L2** | Fixed steps in order, optional human review | Email reply, report generation, data pipeline |
| **L3** | Open-ended, quality must converge | Deep research, code optimization, brainstorming |

## Step 2: Create Tools

For each external capability the Agent needs, create a tool file.

**File location**: `plugins/tools/<domain>_tools.py`

**Template**:

```python
"""<Domain> tools: <brief description>."""

from __future__ import annotations
from langchain_core.tools import tool


@tool
def <tool_name>(<param>: <type>) -> str:
    """<Clear description — the LLM reads this to decide when to call the tool>."""
    # Implementation here
    return result


def get_<domain>_tools() -> list:
    """Return all <domain> tools."""
    return [<tool1>, <tool2>, ...]
```

**Rules**:
- Each `@tool` function MUST have a docstring — the LLM uses it to decide tool selection
- Return `str` (the LLM reads the result as text)
- Keep tools focused: one tool = one action
- Handle errors gracefully: return error messages as strings, don't raise

## Step 3: Register Tools

**3a.** Add to the global registry in `plugins/tools/registry.py`:

```python
# In _populate_defaults(), add:
from plugins.tools.<domain>_tools import get_<domain>_tools
_global_registry.register_many(get_<domain>_tools())
```

**3b.** Enable in `user_config.yaml`:

```yaml
enabled_tools:
  - <tool_name_1>
  - <tool_name_2>
```

## Step 4: Build the Graph

### For L1 — No extra graph needed

L1 uses the built-in ReAct loop. Just register your tools and go:

```python
from core.factory import build_graph
from core.config import AppConfig

config = AppConfig(
    enabled_tools=["your_tool_1", "your_tool_2"],
)
graph = build_graph("l1", config=config)
```

### For L2 — Define Steps

Create `graphs/<agent_name>.py`:

```python
from graphs.l2_workflow import build_l2_graph
from langgraph.checkpoint.memory import MemorySaver

steps = [
    {"name": "step_1_name", "type": "action"},
    {"name": "step_2_name", "type": "approval"},  # Human review
    {"name": "step_3_name", "type": "action"},
]

graph = build_l2_graph(steps, checkpointer=MemorySaver())
```

For L2, each step node is auto-generated. If you need custom logic per step, override `_make_step_node` or create custom node functions — read `graphs/l2_workflow.py` for the pattern.

### For L3 — Define Custom Node Functions

Create `graphs/<agent_name>.py`:

```python
from langchain_core.messages import AIMessage
from graphs.l3_executor import build_l3_graph

def my_plan(state):
    # Planning logic
    return {"messages": [AIMessage(content="Plan: ...")]}

def my_execute(state):
    # Execution logic — call tools, do work
    return {
        "messages": [AIMessage(content="Executed: ...")],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }

def my_evaluate(state):
    # Evaluate quality
    score = ...  # 0.0 to 1.0
    return {
        "messages": [AIMessage(content=f"Quality: {score}")],
        "quality_score": score,
    }

graph = build_l3_graph(
    quality_threshold=0.85,
    max_iterations=5,
    plan_fn=my_plan,
    execute_fn=my_execute,
    evaluate_fn=my_evaluate,
)
```

## Step 5: Create an Entry Point

Create `examples/<agent_name>.py`:

```python
"""<Agent Name>: <one-line description>.

Usage:
    python examples/<agent_name>.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from core.config import AppConfig
from core.factory import build_graph
from core.output import StandardOutput


def main():
    config = AppConfig(
        enabled_tools=[...],
    )
    graph = build_graph("<tier>", config=config)

    result = graph.invoke({
        "messages": [HumanMessage(content="<initial prompt>")],
        "metadata": {},
        "llm_call_count": 0,
        "current_tier": "<tier>",
        # Add tier-specific fields (see below)
    })

    print(StandardOutput.to_markdown(result))


if __name__ == "__main__":
    main()
```

**Tier-specific initial state fields**:

- **L1**: No extra fields needed
- **L2**: Add `"current_step": "", "approval_status": "", "artifacts": {}`
- **L3**: Add `"iteration_count": 0, "reflection_log": [], "quality_score": 0.0, "is_complete": False`

## Step 6: Write Tests

Create `tests/test_<agent_name>.py`:

```python
from tests.conftest import MockLLM
from langchain_core.messages import AIMessage, HumanMessage

class TestMyAgent:
    def test_tools_work(self):
        """Test each tool individually."""
        from plugins.tools.<domain>_tools import <tool>
        result = <tool>.invoke({...})
        assert "expected" in result

    def test_graph_runs(self, monkeypatch):
        """Test the full graph with mock LLM."""
        from graphs import l1_reactor
        mock = MockLLM([AIMessage(content="Mock response")])
        monkeypatch.setattr(l1_reactor, "_get_llm", lambda *a, **kw: mock)
        # ... invoke and assert
```

Then run: `pytest tests/test_<agent_name>.py -v`

## Step 7: Register in langgraph.json (Optional)

If you want the Agent visible in LangGraph Studio:

```json
{
  "graphs": {
    "l1_reactor": "./core/factory.py:l1_graph",
    "l2_workflow": "./core/factory.py:l2_graph",
    "l3_executor": "./core/factory.py:l3_graph",
    "my_agent": "./graphs/<agent_name>.py:graph"
  }
}
```

## Architecture Reference

```
core/
  state.py          — BaseState, L2State, L3State definitions
  config.py         — AppConfig (reads user_config.yaml)
  factory.py        — build_graph("l1"|"l2"|"l3", config=..., **kwargs)
  llm.py            — create_llm("provider:model") — OpenAI/Gemini/Claude/DeepSeek
  safety.py         — BudgetGuard (loop limit), PIISanitizer (PII masking)
  output.py         — StandardOutput.to_markdown() / .to_json()
  runtime_config.py — RuntimeContext (user_id, llm_model, temperature, etc.)

graphs/
  l1_reactor.py     — ReAct loop: llm_call ↔ tool_node, with BudgetGuard
  l2_workflow.py     — Linear DAG: step1 → step2 → ..., interrupt() for approval
  l3_executor.py     — Closed loop: plan → execute → evaluate → reflect → revise

plugins/tools/
  basic_tools.py    — echo_tool, current_time
  file_tools.py     — read_file, write_file, inspect_dir, search_in_files
  registry.py       — ToolRegistry: register, get_many, introductions
  mcp_adapter.py    — MCPAdapter: load tools from MCP servers

plugins/memory/
  sqlite_store.py   — get_sqlite_checkpointer() for state persistence
  user_profile.py   — UserProfile: JSON-based cross-session preferences
```

## Checklist

Before declaring done, verify:

- [ ] Tools created in `plugins/tools/` with clear docstrings
- [ ] Tools registered in `registry.py` `_populate_defaults()`
- [ ] Tools enabled in `user_config.yaml`
- [ ] Graph built using correct tier (L1/L2/L3)
- [ ] Entry point script in `examples/`
- [ ] Tests pass: `pytest tests/test_<agent_name>.py -v`
- [ ] Full regression: `pytest tests/ -v` (all existing tests still pass)
