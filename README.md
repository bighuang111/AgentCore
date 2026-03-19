# Omni-Harness: Universal Agent Development Shell

Omni-Harness 是基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的通用 Agent 开发壳工程。它将编排（Orchestration）、工具（Tools）、记忆（Memory）与监控（Observability）彻底解耦，通过**三阶复杂度分层**覆盖从即问即答到自主闭环的全部任务类型。

```
omni-harness/
├── core/                     # 核心底座（不可变编排逻辑）
│   ├── state.py              # BaseState / L2State / L3State
│   ├── config.py             # AppConfig — YAML 配置驱动
│   ├── runtime_config.py     # RuntimeContext — 运行时注入
│   ├── factory.py            # 图工厂 — build_graph("l1"|"l2"|"l3")
│   ├── safety.py             # BudgetGuard / PIISanitizer
│   └── output.py             # StandardOutput — Markdown / JSON 格式化
├── graphs/                   # 三阶编排模板
│   ├── l1_reactor.py         # L1 原子反应器 — ReAct
│   ├── l2_workflow.py        # L2 标准工作流 — 线性 DAG + 审批
│   └── l3_executor.py        # L3 自主执行器 — 反思闭环
├── plugins/                  # 热插拔功能区
│   ├── tools/                # 工具插件
│   │   ├── basic_tools.py    # echo_tool, current_time
│   │   ├── file_tools.py     # read_file, write_file, inspect_dir, search_in_files
│   │   ├── registry.py       # ToolRegistry — 自发现 + 自介绍
│   │   └── mcp_adapter.py    # MCP 协议动态工具加载
│   ├── memory/               # 记忆插件
│   │   ├── sqlite_store.py   # SQLite / MemorySaver 持久化
│   │   └── user_profile.py   # UserProfile — JSON 跨会话偏好
│   └── observers/            # 观测插件
│       ├── langsmith_tracer.py  # LangSmith 全链路追踪
│       ├── cost_tracker.py      # Token 消耗 & 成本估算
│       └── logger.py            # 结构化 JSON 日志
├── tests/                    # 107 个测试用例
├── langgraph.json            # LangGraph CLI / Studio 入口
├── user_config.yaml          # 用户配置
└── .env                      # 环境变量
```

---

## 目录

- [快速开始](#快速开始)
- [三阶编排架构](#三阶编排架构)
  - [L1 原子反应器](#l1-原子反应器-atomic-reactor)
  - [L2 标准工作流](#l2-标准工作流-standard-workflow)
  - [L3 自主执行器](#l3-自主执行器-autonomous-executor)
  - [如何选择编排层级](#如何选择编排层级)
- [配置系统](#配置系统)
  - [user_config.yaml](#user_configyaml)
  - [AppConfig API](#appconfig-api)
  - [RuntimeContext 运行时注入](#runtimecontext-运行时注入)
- [插件体系](#插件体系)
  - [工具插件](#工具插件)
    - [内置工具](#内置工具)
    - [文件工具](#文件工具)
    - [工具注册表](#工具注册表)
    - [MCP 适配器](#mcp-适配器)
    - [自定义工具开发](#自定义工具开发)
  - [记忆插件](#记忆插件)
    - [SQLite 持久化](#sqlite-持久化)
    - [用户偏好](#用户偏好)
  - [观测插件](#观测插件)
    - [LangSmith 追踪](#langsmith-追踪)
    - [成本追踪](#成本追踪)
    - [结构化日志](#结构化日志)
- [安全网关](#安全网关)
  - [BudgetGuard 预算控制](#budgetguard-预算控制)
  - [PIISanitizer 隐私脱敏](#piisanitizer-隐私脱敏)
- [标准化输出](#标准化输出)
- [Graph Factory API](#graph-factory-api)
- [LangGraph Studio 集成](#langgraph-studio-集成)
- [测试](#测试)
- [高级特性](#高级特性)
  - [RetryPolicy 容错](#retrypolicy-容错)
  - [Command 动态路由](#command-动态路由)
  - [interrupt() 人机协作](#interrupt-人机协作)

---

## 快速开始

### 环境要求

- Python 3.12+（推荐 Anaconda）
- Node.js v22+（LangGraph CLI 需要）

### 安装

```bash
# 克隆项目
cd /path/to/AgentCore

# 安装依赖（含开发工具）
pip install -e ".[dev]"

# 配置环境变量
cp .env .env.local
# 编辑 .env.local，填入真实的 OPENAI_API_KEY
```

### 运行测试

```bash
pytest tests/ -v
# 107 passed
```

### 启动 LangGraph Studio

```bash
langgraph dev
# 浏览器自动打开 Studio UI，可看到 l1_reactor / l2_workflow / l3_executor 三个图
```

### 最小示例：L1 即问即答

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from core.factory import build_graph

# 构建 L1 图
graph = build_graph("l1")

# 调用
result = graph.invoke({
    "messages": [HumanMessage(content="现在几点了？")],
    "metadata": {},
    "llm_call_count": 0,
    "current_tier": "l1",
})
print(result["messages"][-1].content)
```

---

## 三阶编排架构

Omni-Harness 根据任务的**思考深度**（而非职业角色）将编排分为三个标准层级：

```
┌──────────────────────────────────────────────────────────┐
│  L3 自主执行器  │  闭环循环，自我反思，质量收敛           │
├──────────────────────────────────────────────────────────┤
│  L2 标准工作流  │  线性 DAG，SOP 步骤，人工审批           │
├──────────────────────────────────────────────────────────┤
│  L1 原子反应器  │  单跳 ReAct，即问即答                   │
└──────────────────────────────────────────────────────────┘
```

### L1 原子反应器 (Atomic Reactor)

**适用场景**：单跳 ReAct，即问即答——文件格式转换、单次查询、文案翻译。

**图结构**：

```
START → llm_call ──┬──→ END
                   │
            (有 tool_calls?)
                   │
                   ↓
              tool_node → llm_call (循环)
```

**特性**：
- `BudgetGuard` 限制最大循环次数，防止死循环
- `RetryPolicy(max_attempts=3)` 容错：LLM 或工具调用短暂失败自动重试
- `context_schema=RuntimeContext` 支持运行时注入 `user_id`、`llm_model` 等

**State 字段**（`BaseState`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `Annotated[list[AnyMessage], add_messages]` | 消息历史（自动合并） |
| `metadata` | `dict[str, Any]` | 任意元数据 |
| `llm_call_count` | `int` | LLM 调用计数 |
| `current_tier` | `str` | 当前编排层级标识 |

**使用方式**：

```python
from core.factory import build_graph

# 默认配置
graph = build_graph("l1")

# 自定义配置
from core.config import AppConfig, SafetyConfig
config = AppConfig(
    safety=SafetyConfig(max_loops=5),
    enabled_tools=["echo_tool", "read_file"],
)
graph = build_graph("l1", config=config)

# 带运行时上下文调用
result = graph.invoke(
    {
        "messages": [HumanMessage(content="Hello")],
        "metadata": {"source": "api"},
        "llm_call_count": 0,
        "current_tier": "l1",
    },
    context={"user_id": "u_001", "llm_model": "gpt-4o", "temperature": 0.3},
)
```

---

### L2 标准工作流 (Standard Workflow)

**适用场景**：线性 DAG，有明确的 SOP 步骤与审批节点——自动周报、需求文档撰写、竞品对比。

**图结构**：

```
START → step_1 → step_2(approval) → step_3 → ... → END
                      ↑
               interrupt() 暂停
               Command(resume=value) 恢复
```

**特性**：
- 步骤类型 `"action"`（自动执行）和 `"approval"`（人工审批）
- 审批节点使用 `interrupt()` 在节点**内部**暂停，携带上下文提示
- 通过 `Command(resume="approved")` 恢复，审批决策作为返回值流入后续逻辑
- `RetryPolicy` 仅作用于 action 节点（审批节点不重试）

**State 字段**（`L2State` 继承 `BaseState`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `current_step` | `str` | 当前执行步骤名 |
| `approval_status` | `str` | 最近一次审批的决策结果 |
| `artifacts` | `dict[str, Any]` | 各步骤的产出（`步骤名: "completed"/"approved"`） |

**使用方式**：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from graphs.l2_workflow import build_l2_graph

# 定义 SOP 步骤
steps = [
    {"name": "gather_requirements", "type": "action"},
    {"name": "draft_document",      "type": "action"},
    {"name": "manager_review",      "type": "approval"},   # 需人工审批
    {"name": "publish",             "type": "action"},
]

# 构建图（审批需要 checkpointer）
graph = build_l2_graph(steps, checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "workflow-001"}}

# 第一次调用：执行到 manager_review 时暂停
result = graph.invoke({
    "messages": [HumanMessage(content="开始周报流程")],
    "metadata": {},
    "llm_call_count": 0,
    "current_tier": "l2",
    "current_step": "",
    "approval_status": "",
    "artifacts": {},
}, config=config)

print(result["artifacts"])
# {"gather_requirements": "completed", "draft_document": "completed"}
# manager_review 尚未出现 → 被 interrupt() 暂停

# 审批恢复：传入审批决策
result = graph.invoke(Command(resume="approved"), config=config)

print(result["artifacts"])
# {"gather_requirements": "completed", "draft_document": "completed",
#  "manager_review": "approved", "publish": "completed"}
print(result["approval_status"])  # "approved"
```

**多个审批节点**：

```python
steps = [
    {"name": "draft",     "type": "action"},
    {"name": "review_1",  "type": "approval"},   # 第一次暂停
    {"name": "revise",    "type": "action"},
    {"name": "review_2",  "type": "approval"},   # 第二次暂停
    {"name": "publish",   "type": "action"},
]

graph = build_l2_graph(steps, checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "multi-approval"}}

graph.invoke(initial_state, config=config)              # → 暂停在 review_1
graph.invoke(Command(resume="pass"), config=config)     # → 暂停在 review_2
graph.invoke(Command(resume="pass"), config=config)     # → 完成全部
```

---

### L3 自主执行器 (Autonomous Executor)

**适用场景**：带有自我反思与逻辑修正的闭环循环——深度调研、自动化代码优化、脑暴迭代。

**图结构**：

```
START → plan → execute → evaluate ──┬──→ END (质量达标 / 达到上限 / 手动完成)
                 ↑                  │
                 │                  ↓
                 └── revise ← reflect (反思 + 修正)
```

**特性**：
- `evaluate` 节点使用 `Command(goto=...)` 动态路由，合并状态更新和路由决策
- 三重终止条件：`quality_score >= threshold`、`iteration_count >= max_iterations`、`is_complete == True`
- 所有节点均有 `RetryPolicy(max_attempts=3)`
- 5 个节点函数全部支持自定义覆盖

**State 字段**（`L3State` 继承 `BaseState`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `iteration_count` | `int` | 执行-评估循环次数 |
| `reflection_log` | `list[str]` | 反思日志（每次反思追加一条） |
| `quality_score` | `float` | 评估得分（0.0 ~ 1.0） |
| `is_complete` | `bool` | 手动完成标志（优先级最高） |

**使用方式**：

```python
from core.factory import build_graph

# 使用默认节点函数
graph = build_graph("l3", quality_threshold=0.85, max_iterations=8)

result = graph.invoke({
    "messages": [HumanMessage(content="深度调研 AI Agent 架构")],
    "metadata": {},
    "llm_call_count": 0,
    "current_tier": "l3",
    "iteration_count": 0,
    "reflection_log": [],
    "quality_score": 0.0,
    "is_complete": False,
})

print(f"迭代次数: {result['iteration_count']}")
print(f"最终质量: {result['quality_score']}")
print(f"反思日志: {result['reflection_log']}")
```

**自定义节点函数**：

```python
from langchain_core.messages import AIMessage
from graphs.l3_executor import build_l3_graph

def my_evaluate(state):
    """自定义评估逻辑：调用 LLM 评分"""
    # ... 调用 LLM 评估质量 ...
    score = 0.75  # 示例
    return {
        "messages": [AIMessage(content=f"评估得分: {score}")],
        "quality_score": score,
    }

def my_reflect(state):
    """自定义反思逻辑"""
    log = list(state.get("reflection_log", []))
    log.append(f"第{state['iteration_count']}轮反思：质量不足，需要补充数据源")
    return {
        "messages": [AIMessage(content="反思完成")],
        "reflection_log": log,
    }

graph = build_l3_graph(
    quality_threshold=0.9,
    max_iterations=5,
    evaluate_fn=my_evaluate,
    reflect_fn=my_reflect,
)
```

**L3 `build_l3_graph` 全部参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `quality_threshold` | `float` | `0.8` | 质量达标阈值 |
| `max_iterations` | `int` | `5` | 最大迭代次数 |
| `plan_fn` | `Callable` | 内置 | 自定义计划节点 |
| `execute_fn` | `Callable` | 内置 | 自定义执行节点 |
| `evaluate_fn` | `Callable` | 内置 | 自定义评估节点 |
| `reflect_fn` | `Callable` | 内置 | 自定义反思节点 |
| `revise_fn` | `Callable` | 内置 | 自定义修正节点 |

---

### 如何选择编排层级

```
你的任务需要多少步？
│
├─ 1步 → 能用一次 LLM + 工具解决吗？
│        ├─ 是 → L1 原子反应器
│        └─ 否 → L2 标准工作流
│
├─ 2~N步 → 步骤是否固定（SOP）？
│           ├─ 是 → L2 标准工作流
│           │       └─ 有审批节点？→ 加 type="approval"
│           └─ 否 → L3 自主执行器
│
└─ 不确定 → 需要 AI 自己判断何时完成？
             ├─ 是 → L3 自主执行器
             └─ 否 → L2 标准工作流
```

| 判断维度 | L1 | L2 | L3 |
|---------|----|----|-----|
| 步骤数 | 1 | 2~N（固定） | 不确定 |
| 是否需要审批 | 否 | 支持 | 可选 |
| 是否需要反思 | 否 | 否 | 是 |
| 典型耗时 | 秒级 | 分钟级 | 分钟~小时 |
| Token 消耗 | 低 | 中 | 高 |
| 典型场景 | 翻译、查询、格式转换 | 周报、文档、对比表 | 调研、代码优化、脑暴 |

---

## 配置系统

### user_config.yaml

项目根目录下的 `user_config.yaml` 是用户级配置文件，驱动图工厂的构建行为：

```yaml
# 默认 LLM 标识（格式：provider:model）
default_llm: "openai:gpt-4o-mini"

# 安全控制
safety:
  max_loops: 10      # L1 最大 ReAct 循环次数
  max_tokens: 4096   # Token 预算上限

# 启用的编排层级
enabled_tiers:
  - l1
  - l2
  - l3

# 启用的工具（对应 ToolRegistry 中的名称）
enabled_tools:
  - echo_tool
  - current_time
  - read_file
  - write_file
  - inspect_dir
  - search_in_files
```

### AppConfig API

```python
from core.config import AppConfig, SafetyConfig

# 方式 1：从 YAML 文件加载
config = AppConfig.from_yaml("user_config.yaml")

# 方式 2：从 YAML 加载（文件不存在则返回默认值）
config = AppConfig.from_yaml("nonexistent.yaml")  # 不报错，返回默认

# 方式 3：代码中直接构造
config = AppConfig(
    default_llm="openai:gpt-4o",
    safety=SafetyConfig(max_loops=5, max_tokens=2048),
    enabled_tiers=["l1", "l2", "l3"],
    enabled_tools=["echo_tool", "read_file"],
)

# 传入 factory
from core.factory import build_graph
graph = build_graph("l1", config=config)
```

**AppConfig 字段**：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `default_llm` | `str` | `"openai:gpt-4o-mini"` | 默认 LLM 标识 |
| `safety.max_loops` | `int` | `10` | L1 最大循环次数（`≥1`） |
| `safety.max_tokens` | `int` | `4096` | Token 预算上限（`≥1`） |
| `enabled_tiers` | `list[str]` | `["l1"]` | 启用的编排层级 |
| `enabled_tools` | `list[str]` | `["echo_tool", "current_time"]` | 启用的工具列表 |

### RuntimeContext 运行时注入

除了构建时的静态配置，每次 `invoke` 调用可以注入**运行时上下文**：

```python
result = graph.invoke(
    state,
    context={
        "user_id": "u_001",          # 当前用户标识
        "llm_model": "gpt-4o",       # 覆盖默认模型
        "temperature": 0.3,           # LLM 温度
        "language": "zh-CN",          # 语言偏好
        "session_id": "sess_abc123",  # 会话标识
    },
)
```

节点函数中通过 `runtime` 参数获取：

```python
def my_node(state: BaseState, *, runtime=None) -> dict:
    ctx = runtime.context if runtime else {}
    user_id = ctx.get("user_id", "anonymous")
    model = ctx.get("llm_model", "gpt-4o-mini")
    # ...
```

**RuntimeContext 字段**（全部可选）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | `str` | 当前用户标识，用于记忆隔离 |
| `llm_model` | `str` | 覆盖默认 LLM 模型 |
| `temperature` | `float` | LLM 采样温度 |
| `language` | `str` | 语言偏好（`"zh-CN"`, `"en"` 等） |
| `session_id` | `str` | 会话标识，用于追踪 |

---

## 插件体系

所有插件位于 `plugins/` 目录，按功能分为 **tools**（工具）、**memory**（记忆）、**observers**（观测）三类。插件通过配置文件和注册表实现热插拔。

### 工具插件

#### 内置工具

`plugins/tools/basic_tools.py` — 开箱即用的演示工具：

| 工具 | 签名 | 说明 |
|------|------|------|
| `echo_tool` | `(text: str) -> str` | 原样返回输入文本，前缀 `"Echo: "` |
| `current_time` | `() -> str` | 返回当前 UTC 时间（ISO 8601 格式） |

```python
from plugins.tools.basic_tools import echo_tool, current_time, get_basic_tools

echo_tool.invoke({"text": "hello"})    # "Echo: hello"
current_time.invoke({})                 # "2026-03-19T06:30:00+00:00"
get_basic_tools()                       # [echo_tool, current_time]
```

#### 文件工具

`plugins/tools/file_tools.py` — 文件系统操作四件套：

| 工具 | 签名 | 说明 |
|------|------|------|
| `read_file` | `(file_path: str) -> str` | 读取文件内容（UTF-8）。文件不存在返回错误信息 |
| `write_file` | `(file_path: str, content: str) -> str` | 写入文件。自动创建父目录。返回写入字节数 |
| `inspect_dir` | `(dir_path: str) -> str` | 列出目录内容，显示 `[DIR]`/`[FILE]`、文件名、大小 |
| `search_in_files` | `(dir_path: str, pattern: str) -> str` | 递归搜索目录中所有文件的文本内容，返回最多 50 条匹配（`文件:行号: 内容`） |

```python
from plugins.tools.file_tools import read_file, write_file, inspect_dir, search_in_files

read_file.invoke({"file_path": "/tmp/data.txt"})
write_file.invoke({"file_path": "/tmp/out.txt", "content": "hello"})
inspect_dir.invoke({"dir_path": "/tmp"})
search_in_files.invoke({"dir_path": "/tmp", "pattern": "hello"})
```

#### 工具注册表

`plugins/tools/registry.py` — 中央工具注册表，支持自发现、自介绍、按配置加载。

```python
from plugins.tools.registry import ToolRegistry, get_global_registry

# 全局注册表（首次访问自动加载 basic_tools + file_tools）
registry = get_global_registry()

# 查看所有已注册工具
registry.list_names()
# ['echo_tool', 'current_time', 'read_file', 'write_file', 'inspect_dir', 'search_in_files']

# 按名称获取单个工具
tool = registry.get("read_file")

# 按名称批量获取（跳过不存在的）
tools = registry.get_many(["echo_tool", "read_file", "nonexistent"])
# [echo_tool, read_file]

# 获取所有工具
all_tools = registry.all_tools()

# 生成工具自介绍（Markdown 格式，可直接注入 System Prompt）
print(registry.introductions())
# - **echo_tool**: Echo back the input text. Useful for testing.
# - **current_time**: Return the current UTC time in ISO format.
# - **read_file**: Read and return the contents of a file.
# ...
```

**注册自定义工具**：

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """Search internal knowledge base."""
    return f"Results for: {query}"

registry = get_global_registry()
registry.register(my_custom_tool)

# 现在可以在 user_config.yaml 中启用
# enabled_tools:
#   - my_custom_tool
```

**`ToolRegistry` API 完整列表**：

| 方法 | 说明 |
|------|------|
| `register(tool_obj)` | 注册单个工具（按 `tool.name` 属性） |
| `register_many(tools)` | 批量注册 |
| `get(name) -> tool \| None` | 按名称获取 |
| `get_many(names) -> list` | 批量获取（跳过不存在的） |
| `all_tools() -> list` | 返回所有已注册工具 |
| `list_names() -> list[str]` | 返回所有工具名称 |
| `introductions() -> str` | 生成 Markdown 格式的工具自介绍 |

#### MCP 适配器

`plugins/tools/mcp_adapter.py` — 通过 [Model Context Protocol](https://modelcontextprotocol.io/) 动态加载外部服务器提供的工具。

```python
from plugins.tools.mcp_adapter import MCPAdapter

# 配置 stdio 类型的 MCP 服务器
adapter = MCPAdapter(server_configs=[
    {
        "type": "stdio",
        "name": "my_mcp_server",
        "command": "npx",
        "args": ["-y", "@my-org/mcp-server"],
    },
])

# 加载工具（同步）
tools = adapter.load_tools()

# 或异步加载
tools = await adapter._load_tools_async()

# 检查状态
adapter.is_loaded   # True
adapter.tools       # [tool1, tool2, ...]
```

**支持的服务器类型**：

| 类型 | 配置字段 | 说明 |
|------|---------|------|
| `stdio` | `command`, `args`, `env` | 通过子进程通信 |
| `sse` | `url` | 通过 HTTP SSE 通信 |

**SSE 示例**：

```python
adapter = MCPAdapter(server_configs=[
    {
        "type": "sse",
        "name": "remote_server",
        "url": "http://localhost:8080/sse",
    },
])
```

#### 自定义工具开发

开发新工具只需三步：

**第一步**：创建工具文件

```python
# plugins/tools/web_tools.py
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # 实现搜索逻辑
    return f"Search results for: {query}"

@tool
def web_browse(url: str) -> str:
    """Browse a webpage and extract text content."""
    # 实现浏览逻辑
    return f"Content from: {url}"

def get_web_tools() -> list:
    return [web_search, web_browse]
```

**第二步**：注册到全局注册表

```python
# plugins/tools/registry.py 的 _populate_defaults() 中添加：
from plugins.tools.web_tools import get_web_tools
_global_registry.register_many(get_web_tools())
```

**第三步**：在配置中启用

```yaml
# user_config.yaml
enabled_tools:
  - echo_tool
  - web_search
  - web_browse
```

---

### 记忆插件

#### SQLite 持久化

`plugins/memory/sqlite_store.py` — 为 LangGraph 图提供状态持久化（Checkpointer）。

```python
from plugins.memory.sqlite_store import get_sqlite_checkpointer

# 方式 1：内存存储（进程结束即丢失，适合开发/测试）
saver = get_sqlite_checkpointer()

# 方式 2：文件存储（持久化到磁盘，适合生产）
saver = get_sqlite_checkpointer("./data/checkpoints.db")

# 用于图编译
from graphs.l2_workflow import build_l2_graph
graph = build_l2_graph(steps=steps, checkpointer=saver)

# 用于支持 interrupt() 和 time-travel
config = {"configurable": {"thread_id": "session-001"}}
result = graph.invoke(state, config=config)
```

#### 用户偏好

`plugins/memory/user_profile.py` — 跨会话的用户偏好存储（JSON 文件）。

```python
from plugins.memory.user_profile import UserProfile

# 创建 / 加载用户偏好（自动持久化到文件）
profile = UserProfile("./data/user_001.json")

# 设置偏好（立即写入文件）
profile.set("writing_style", "formal")
profile.set("preferred_language", "zh-CN")
profile.set("tech_stack", ["Python", "React", "PostgreSQL"])

# 读取偏好
style = profile.get("writing_style")                    # "formal"
lang = profile.get("preferred_language", "en")           # "zh-CN"
unknown = profile.get("nonexistent", "default_value")    # "default_value"

# 删除偏好
profile.delete("tech_stack")

# 获取全部偏好
all_prefs = profile.all()
# {"writing_style": "formal", "preferred_language": "zh-CN"}
```

**`UserProfile` API**：

| 方法 | 说明 |
|------|------|
| `get(key, default=None)` | 读取，不存在返回默认值 |
| `set(key, value)` | 写入并立即持久化 |
| `delete(key)` | 删除并立即持久化 |
| `all() -> dict` | 返回全部偏好的副本 |

---

### 观测插件

#### LangSmith 追踪

`plugins/observers/langsmith_tracer.py` — 一键开启/关闭 [LangSmith](https://smith.langchain.com/) 全链路追踪。

```python
from plugins.observers.langsmith_tracer import configure_langsmith, is_tracing_enabled

# 开启追踪（设置环境变量 LANGCHAIN_TRACING_V2=true）
configure_langsmith(project="my-agent-project", enabled=True)

# 关闭追踪
configure_langsmith(enabled=False)

# 自动检测（读取 .env 中的 LANGCHAIN_TRACING_V2）
configure_langsmith(project="omni-harness")

# 检查当前状态
is_tracing_enabled()  # True / False
```

开启后，所有 LangGraph 图的执行都会自动上报到 LangSmith，可在 Dashboard 中查看：
- 每个节点的输入/输出
- LLM 调用的 Token 消耗
- 工具调用的参数和返回值
- 完整的执行时间线

#### 成本追踪

`plugins/observers/cost_tracker.py` — 跟踪 Token 消耗并估算成本。

```python
from plugins.observers.cost_tracker import CostTracker

tracker = CostTracker()

# 记录 LLM 调用
tracker.record(model="gpt-4o-mini", prompt_tokens=500, completion_tokens=200)
tracker.record(model="gpt-4o", prompt_tokens=1000, completion_tokens=500)

# 查看统计
tracker.total_calls                # 2
tracker.total_tokens               # 2200
tracker.estimated_cost()           # 0.006575 (USD)

# 获取完整摘要
tracker.summary()
# {
#     "total_calls": 2,
#     "total_tokens": 2200,
#     "estimated_cost_usd": 0.006575,
#     "records": [
#         {"model": "gpt-4o-mini", "prompt": 500, "completion": 200},
#         {"model": "gpt-4o", "prompt": 1000, "completion": 500},
#     ],
# }
```

**内置定价表**（USD / 1K tokens）：

| 模型 | 输入价 | 输出价 |
|------|--------|--------|
| `gpt-4o` | $0.0025 | $0.01 |
| `gpt-4o-mini` | $0.00015 | $0.0006 |
| 其他模型 | $0.001（默认） | $0.002（默认） |

#### 结构化日志

`plugins/observers/logger.py` — JSON Lines 格式的结构化日志。

```python
from plugins.observers.logger import get_logger

logger = get_logger("my-agent")

logger.info("Agent started")
logger.warning("Token budget at 80%")
logger.error("Tool call failed", exc_info=True)
```

输出格式（每行一个 JSON 对象）：

```json
{"timestamp": "2026-03-19T06:30:00+00:00", "level": "INFO", "logger": "my-agent", "message": "Agent started"}
{"timestamp": "2026-03-19T06:30:01+00:00", "level": "WARNING", "logger": "my-agent", "message": "Token budget at 80%"}
{"timestamp": "2026-03-19T06:30:02+00:00", "level": "ERROR", "logger": "my-agent", "message": "Tool call failed", "exception": "ConnectionError(...)"}
```

---

## 安全网关

### BudgetGuard 预算控制

`core/safety.py` — 防止 L1 ReAct 循环陷入无限 LLM 调用。

```python
from core.safety import BudgetGuard

guard = BudgetGuard(max_loops=5)

# 在图的条件边中使用
# 当 llm_call_count >= max_loops 时强制返回 "__end__"
# 否则检查最后一条消息是否有 tool_calls：有则返回 "tool_node"，无则 "__end__"
result = guard.should_continue(state)  # "tool_node" | "__end__"
```

**工作原理**：

```
llm_call_count < max_loops AND 有 tool_calls  →  "tool_node"（继续）
llm_call_count >= max_loops                    →  "__end__"（强制终止）
无 tool_calls                                  →  "__end__"（正常结束）
```

### PIISanitizer 隐私脱敏

`core/safety.py` — 双向过滤敏感个人身份信息。

```python
from core.safety import PIISanitizer

# 默认启用所有模式
sanitizer = PIISanitizer()

text = "联系人: john@example.com, 电话: 555-123-4567, SSN: 123-45-6789"
clean = sanitizer.sanitize(text)
# "联系人: [EMAIL], 电话: [PHONE], SSN: [SSN]"

# 仅启用部分模式
sanitizer = PIISanitizer(enabled_patterns=["email", "phone"])
clean = sanitizer.sanitize(text)
# "联系人: [EMAIL], 电话: [PHONE], SSN: 123-45-6789"
```

**支持的 PII 模式**：

| 模式名 | 匹配内容 | 替换为 |
|--------|---------|--------|
| `email` | `user@domain.com` | `[EMAIL]` |
| `phone` | `555-123-4567`、`(555)123-4567`、`+1-555-123-4567` | `[PHONE]` |
| `ssn` | `123-45-6789` | `[SSN]` |
| `credit_card` | `4111-1111-1111-1111`、`4111 1111 1111 1111` | `[CREDIT_CARD]` |
| `ip_address` | `192.168.1.1` | `[IP_ADDRESS]` |

**配合 StandardOutput 使用**：

```python
from core.output import StandardOutput
from core.safety import PIISanitizer

result = graph.invoke(state)
md = StandardOutput.to_markdown(result)
clean_md = PIISanitizer().sanitize(md)
```

---

## 标准化输出

`core/output.py` — 确保所有层级的 Agent 输出统一格式化。

```python
from core.output import StandardOutput

result = graph.invoke(state)

# Markdown 格式
md = StandardOutput.to_markdown(result)
# ## Agent Output (Tier: l1)
#
# 这是 Agent 的回答内容...
#
# ---
# _LLM calls: 3_

# JSON 结构
json_out = StandardOutput.to_json(result)
# {
#     "tier": "l1",
#     "output": "这是 Agent 的回答内容...",
#     "llm_call_count": 3,
#     "metadata": {"source": "api"}
# }

# JSON 字符串（可直接返回给前端或写入文件）
json_str = StandardOutput.to_json_string(result)
```

---

## Graph Factory API

`core/factory.py` 提供统一的 `build_graph()` 入口：

```python
from core.factory import build_graph
from core.config import AppConfig

# L1：自动从 user_config.yaml 加载配置
graph = build_graph("l1")

# L1：指定配置
graph = build_graph("l1", config=AppConfig(safety=SafetyConfig(max_loops=3)))

# L2：必须传入 steps
graph = build_graph("l2", steps=[
    {"name": "draft", "type": "action"},
    {"name": "review", "type": "approval"},
    {"name": "publish", "type": "action"},
])

# L3：可自定义阈值和节点函数
graph = build_graph("l3",
    quality_threshold=0.9,
    max_iterations=10,
    evaluate_fn=my_evaluate,
    reflect_fn=my_reflect,
)

# 不存在的 tier 会抛出 ValueError
build_graph("l99")  # ValueError: Unknown tier: l99. Available: ['l1', 'l2', 'l3']
```

**各 tier 支持的 kwargs**：

| Tier | 参数 | 说明 |
|------|------|------|
| `l1` | （无额外参数） | 从 config 读取 `max_loops` 和 `enabled_tools` |
| `l2` | `steps: list[dict]` | 步骤列表（必传） |
| `l3` | `quality_threshold: float` | 质量阈值 |
| `l3` | `max_iterations: int` | 最大迭代次数 |
| `l3` | `plan_fn`, `execute_fn`, `evaluate_fn`, `reflect_fn`, `revise_fn` | 自定义节点函数 |

---

## LangGraph Studio 集成

`langgraph.json` 注册了三个图供 Studio UI 可视化和交互：

```json
{
  "dependencies": [".", "langchain_openai"],
  "graphs": {
    "l1_reactor":  "./core/factory.py:l1_graph",
    "l2_workflow": "./core/factory.py:l2_graph",
    "l3_executor": "./core/factory.py:l3_graph"
  },
  "env": "./.env",
  "python_version": "3.12"
}
```

启动：

```bash
langgraph dev
# 默认地址：http://127.0.0.1:2024
# Studio UI 自动打开，可看到三个图的拓扑、节点、边
```

---

## 测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定阶段
pytest tests/test_phase1.py -v    # 骨架 + L1
pytest tests/test_phase2.py -v    # 安全 + 配置
pytest tests/test_phase3.py -v    # L2 + 审批
pytest tests/test_phase4.py -v    # L3 + 反思
pytest tests/test_phase5.py -v    # 插件体系
pytest tests/test_phase6.py -v    # 监控 + 输出

# 运行增强功能测试
pytest tests/test_enhancements.py -v  # RetryPolicy / Command / interrupt / context

# 运行集成测试
pytest tests/test_integration.py -v   # 全链路 L1/L2/L3
```

**测试架构**：

- 所有测试默认使用 **Mock LLM**（`tests/conftest.py` 中的 `MockLLM` 类），无需真实 API Key
- `mock_llm_simple` fixture：返回纯文本回复
- `mock_llm_with_tool_call` fixture：先调用工具，再返回最终回答
- 107 个测试用例，覆盖所有模块

---

## 高级特性

### RetryPolicy 容错

所有 tier 的关键节点都配置了 `RetryPolicy(max_attempts=3)`，自动处理瞬态故障：

```python
from langgraph.types import RetryPolicy

# L1: llm_call 和 tool_node 均有 retry
# L2: action 节点有 retry，approval 节点无 retry（避免重复审批）
# L3: plan/execute/evaluate/reflect/revise 全部有 retry

# 自定义 retry 策略
from langgraph.graph import StateGraph, START, END

graph = StateGraph(BaseState)
graph.add_node(
    "my_node",
    my_function,
    retry_policy=RetryPolicy(
        max_attempts=5,              # 最多重试 5 次
        initial_interval=1.0,        # 初始等待 1 秒
        backoff_factor=2.0,          # 指数退避倍数
        max_interval=10.0,           # 最大等待 10 秒
    ),
)
```

### Command 动态路由

L3 的 `evaluate` 节点使用 `Command(goto=...)` 合并状态更新和路由决策：

```python
from langgraph.types import Command
from langgraph.graph import END

def evaluate_and_route(state):
    score = evaluate_quality(state)

    if score >= threshold:
        # 状态更新 + 路由到 END，一步完成
        return Command(
            update={"quality_score": score, "messages": [...]},
            goto=END,
        )
    else:
        return Command(
            update={"quality_score": score, "messages": [...]},
            goto="reflect",
        )
```

对比传统 `add_conditional_edges` 方式：
- **之前**：evaluate 只更新状态 → 外部路由函数决定下一步 → 两处逻辑要对齐
- **现在**：evaluate 同时更新状态和决定路由 → 单一职责，无需 path_map

### interrupt() 人机协作

L2 审批节点使用 `interrupt()` 在节点**内部**暂停（而非 `interrupt_before` 在节点外部）：

```python
from langgraph.types import interrupt, Command

def approval_node(state):
    # 暂停并向用户展示提示信息
    decision = interrupt("请审批此文档。内容摘要: ...")

    # decision 就是 Command(resume=value) 中的 value
    return {
        "approval_status": decision,     # "approved" / "rejected" / 任意字符串
        "artifacts": {...},
    }

# 恢复执行
graph.invoke(Command(resume="approved"), config=config)
```

**优势**：
- 审批节点可以携带上下文提示信息（`interrupt("请审批此文档")`）
- 审批者的回复直接作为返回值参与后续逻辑
- 支持多种审批结果（不仅是 yes/no）
- 支持同一流程中多个审批节点（顺序暂停、顺序恢复）

---

## 技术栈

| 组件 | 版本 |
|------|------|
| Python | 3.12+ |
| LangGraph | ≥ 1.1.3 |
| LangChain Core | ≥ 0.3.0 |
| LangChain OpenAI | ≥ 0.3.0 |
| langchain-mcp-adapters | ≥ 0.2.2 |
| Pydantic | ≥ 2.12.0 |
| PyYAML | ≥ 6.0 |
| Node.js | v22+（LangGraph CLI） |

---

## License

MIT
