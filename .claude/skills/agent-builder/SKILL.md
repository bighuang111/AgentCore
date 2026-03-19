---
name: agent-builder
description: Build custom AI Agents on the Omni-Harness framework. Use when user asks to create, build, develop, or implement any Agent, bot, assistant, or automated workflow. Covers tool creation, graph orchestration (L1/L2/L3), config registration, and testing.
argument-hint: [agent description in natural language]
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, Agent
---

# Omni-Harness Agent Builder

You are building an Agent on top of the **Omni-Harness** framework. This project provides three orchestration tiers, a pluggable tool system, safety guardrails, and multi-provider LLM support.

**Before writing any code**, read this skill completely. Then follow the workflow step by step.

---

## CRITICAL: Project Isolation Rule

**All user Agent code MUST be created inside `workspace/` as an independent project.**

The shell framework (`core/`, `graphs/`, `plugins/`) is READ-ONLY. Never modify files outside `workspace/` when building a user Agent.

### Project structure for each Agent:

```
workspace/
└── <agent-name>/              ← One directory per Agent project
    ├── tools/                 ← Agent-specific tools
    │   └── <domain>_tools.py
    ├── web/                   ← (if web form chosen)
    │   ├── app.py             ← Flask/FastAPI server
    │   ├── templates/
    │   └── static/
    ├── ui/                    ← (if desktop form chosen)
    │   └── main_window.py
    ├── graph.py               ← Graph definition (import from core/graphs)
    ├── config.yaml            ← Agent-specific config
    ├── main.py                ← Entry point
    ├── tests/
    │   └── test_<agent>.py
    ├── requirements.txt       ← Extra dependencies (if any)
    └── README.md              ← Usage instructions
```

### How to import the shell framework:

All workspace projects import from the root package. The entry point `main.py` must add the project root to `sys.path`:

```python
import sys
from pathlib import Path

# Add project root so we can import core/, graphs/, plugins/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now these work:
from core.factory import build_graph
from core.config import AppConfig
from core.llm import create_llm
from core.output import StandardOutput
from graphs.l1_reactor import build_l1_graph
from graphs.l2_workflow import build_l2_graph
from graphs.l3_executor import build_l3_graph
from plugins.tools.registry import ToolRegistry
```

---

## Step 0: Understand the Request

Parse `$ARGUMENTS` (the user's natural language description) and determine:

1. **What the Agent should do** — the core task
2. **What external services it needs** — APIs, email, databases, web scraping, etc.
3. **Whether human approval is needed** — sending emails, publishing, payments, etc.
4. **Whether iteration/quality is needed** — research, writing, optimization loops

## Step 0.5: Environment Setup (MUST DO BEFORE CODING)

Before writing any Agent code, ensure the user's environment is ready. Complete these three sub-steps interactively:

### A. Install Dependencies

The framework requires base + provider-specific packages. Run:

```bash
# Base dependencies (required)
pip install -e ".[dev]"

# If the Agent needs extra packages (e.g., requests, beautifulsoup4),
# install them too and record in workspace/<agent-name>/requirements.txt
```

If the Agent uses a specific LLM provider, also install the provider package:

| Provider | Install Command |
|----------|----------------|
| Google Gemini | `pip install -e ".[google]"` |
| Anthropic Claude | `pip install -e ".[anthropic]"` |
| All providers | `pip install -e ".[all-providers]"` |
| OpenAI (default) | Included in base install |

### B. Configure API Key

Check `.env` in the project root. If the required API key is missing or placeholder, **ask the user to provide it**. Do NOT proceed with coding until at least one valid LLM API key is configured.

Available providers and their env vars:

| Provider | Env Variable | Model Examples |
|----------|-------------|----------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o`, `gpt-4o-mini`, `o3-mini` |
| Google Gemini | `GOOGLE_API_KEY` | `google_genai:gemini-2.5-flash`, `google_genai:gemini-2.5-pro` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic:claude-sonnet-4-6`, `anthropic:claude-haiku-4-5-20251001` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek:deepseek-chat`, `deepseek:deepseek-reasoner` |

If the Agent needs additional service credentials (e.g., `EMAIL_PASSWORD`, `SLACK_TOKEN`), add them to `.env` as well and document them in the Agent's README.

### C. Choose LLM Model

Ask the user which LLM model they want the Agent to use. Default to whatever provider the user already has an API key for. Use the model format `provider:model-name` (e.g., `google_genai:gemini-2.5-flash`).

Write the chosen model into:
1. `workspace/<agent-name>/config.yaml` → `default_llm` field
2. The Agent's code where `create_llm()` is called

**Do NOT skip Step 0.5.** An Agent without dependencies or API keys will fail at runtime.

---

## Step 0.6: Choose Runtime Form

Ask the user: **"以什么形式运行？"** (What runtime form?)

| Option | Description | Action |
|--------|-------------|--------|
| **Terminal** | CLI / command-line (default) | Continue with current flow, `main.py` as entry point |
| **Web** | Browser-based UI (Flask/FastAPI + frontend) | Invoke `frontend-design` skill for UI design, add `web/` dir to project structure |
| **Desktop** | Native desktop app (e.g., Electron, Tkinter, PyQt) | Invoke `frontend-design` skill for UI design, add desktop framework deps |
| **Other** | Custom form — user describes | Adapt based on user description |

### If **web** or **desktop** is selected:

1. Use the `Skill` tool to invoke the `frontend-design` skill, passing the Agent's purpose and required UI elements as context
2. Add UI-related files to the project structure:
   - Web: `workspace/<agent-name>/web/` (Flask/FastAPI server, templates, static assets)
   - Desktop: `workspace/<agent-name>/ui/` (desktop framework files)
3. Add frontend dependencies to `requirements.txt` (e.g., `flask`, `fastapi[standard]`, `pyqt6`, etc.)
4. The Agent's `main.py` should start both the backend graph **and** the UI server/app

### If **terminal** is selected:

No change to existing flow — proceed directly to Step 1.

---

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

## Step 2: Create the Agent Project

```bash
mkdir -p workspace/<agent-name>/tools
mkdir -p workspace/<agent-name>/tests
```

## Step 3: Create Tools

For each external capability the Agent needs, create a tool file.

**File location**: `workspace/<agent-name>/tools/<domain>_tools.py`

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

## Step 4: Create Agent Config

**File**: `workspace/<agent-name>/config.yaml`

```yaml
agent_name: "<agent-name>"
tier: "l1"  # or "l2" or "l3"
default_llm: "google_genai:gemini-3-flash-preview"
runtime_form: "terminal"  # terminal | web | desktop | other

safety:
  max_loops: 10
  max_tokens: 4096

# L2-specific: SOP steps
steps:
  - name: "step_1"
    type: "action"
  - name: "step_2"
    type: "approval"
  - name: "step_3"
    type: "action"

# L3-specific: quality control
quality_threshold: 0.85
max_iterations: 5
```

## Step 5: Build the Graph

**File**: `workspace/<agent-name>/graph.py`

### For L1 — ReAct with custom tools

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.config import AppConfig, SafetyConfig
from graphs.l1_reactor import build_l1_graph
from plugins.tools.basic_tools import get_basic_tools
from tools.<domain>_tools import get_<domain>_tools


def build_agent():
    all_tools = get_basic_tools() + get_<domain>_tools()
    return build_l1_graph(max_loops=10, tools=all_tools)


graph = build_agent()
```

### For L2 — Linear workflow with approval

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langgraph.checkpoint.memory import MemorySaver
from graphs.l2_workflow import build_l2_graph

steps = [
    {"name": "step_1_name", "type": "action"},
    {"name": "step_2_name", "type": "approval"},  # Human review
    {"name": "step_3_name", "type": "action"},
]


def build_agent():
    return build_l2_graph(steps, checkpointer=MemorySaver())


graph = build_agent()
```

### For L3 — Autonomous with reflection

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.messages import AIMessage
from graphs.l3_executor import build_l3_graph


def my_plan(state):
    return {"messages": [AIMessage(content="Plan: ...")]}

def my_execute(state):
    return {
        "messages": [AIMessage(content="Executed: ...")],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }

def my_evaluate(state):
    score = ...  # 0.0 to 1.0
    return {
        "messages": [AIMessage(content=f"Quality: {score}")],
        "quality_score": score,
    }


def build_agent():
    return build_l3_graph(
        quality_threshold=0.85,
        max_iterations=5,
        plan_fn=my_plan,
        execute_fn=my_execute,
        evaluate_fn=my_evaluate,
    )


graph = build_agent()
```

## Step 6: Create Entry Point

**File**: `workspace/<agent-name>/main.py`

```python
"""<Agent Name>: <one-line description>.

Usage:
    python workspace/<agent-name>/main.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Also add the agent's own directory for local tool imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.messages import HumanMessage
from core.output import StandardOutput
from graph import build_agent


def main():
    graph = build_agent()

    result = graph.invoke({
        "messages": [HumanMessage(content="<initial prompt or user input>")],
        "metadata": {},
        "llm_call_count": 0,
        "current_tier": "<tier>",
        # L2 extra: "current_step": "", "approval_status": "", "artifacts": {},
        # L3 extra: "iteration_count": 0, "reflection_log": [], "quality_score": 0.0, "is_complete": False,
    })

    print(StandardOutput.to_markdown(result))


if __name__ == "__main__":
    main()
```

## Step 7: Write Tests

**File**: `workspace/<agent-name>/tests/test_<agent>.py`

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import MockLLM
from langchain_core.messages import AIMessage, HumanMessage


class TestMyAgentTools:
    def test_tool_works(self):
        """Test each tool individually."""
        from tools.<domain>_tools import <tool>
        result = <tool>.invoke({...})
        assert "expected" in result


class TestMyAgentGraph:
    def test_graph_builds(self):
        """Test the graph compiles without error."""
        from graph import build_agent
        graph = build_agent()
        assert graph is not None
```

Run from project root:

```bash
python -m pytest workspace/<agent-name>/tests/ -v
```

## Step 8: Create README

**File**: `workspace/<agent-name>/README.md`

Briefly describe:
- What the Agent does
- How to run it
- What API keys / services are needed
- Example usage

---

## Architecture Reference (READ-ONLY — do not modify)

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

- [ ] **Dependencies installed** — `pip install -e ".[dev]"` + provider extras + agent-specific packages
- [ ] **API key configured** — at least one LLM provider key in `.env`, plus any service-specific credentials
- [ ] **LLM model chosen** — user confirmed which model to use, written to config.yaml and code
- [ ] **Runtime form chosen** — user selected terminal/web/desktop/other
- [ ] If web/desktop: UI designed via `frontend-design` skill and files created
- [ ] Agent project created in `workspace/<agent-name>/`
- [ ] NO files modified outside `workspace/` (shell framework is read-only, except `.env` for credentials)
- [ ] Tools created in `workspace/<agent-name>/tools/` with clear docstrings
- [ ] Graph built in `workspace/<agent-name>/graph.py` using correct tier
- [ ] Config created in `workspace/<agent-name>/config.yaml`
- [ ] Entry point at `workspace/<agent-name>/main.py` — runnable
- [ ] Tests in `workspace/<agent-name>/tests/` — passing
- [ ] README.md with usage instructions (including required env vars and install steps)
- [ ] `requirements.txt` lists any extra dependencies beyond the base framework
- [ ] Smoke test: `python workspace/<agent-name>/main.py` runs without error
