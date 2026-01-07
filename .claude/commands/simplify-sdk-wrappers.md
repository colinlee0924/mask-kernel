# Simplify SDK Wrappers - 執行計劃

## 目標

重構 MASK Kernel SDK，讓開發者可以直接使用原生 LangChain/A2A SDK，MASK 只提供：
1. **SkillMiddleware** - 核心價值（Progressive Disclosure）
2. **Helpers** - 轉換函數（不是封裝）
3. **Observability** - 封裝追蹤設定

## 設計原則

- 讓使用者直接用 `from langchain.agents import create_agent`
- A2A 和 MCP 變成 helpers（`mask.a2a.helpers`, `mask.mcp.helpers`）
- CLI 模板文件化（使用 Jinja2，已是現有依賴）
- 保留 `MaskA2AServer` 和 `SimpleAgent` 但標記為 legacy

---

## Phase 1：建立 Helpers 模組

### 1.1 建立 MCP Helpers

**新增檔案**: `src/mask/mcp/helpers.py`

```python
"""MCP Helper Functions.

這些是 helper functions，回傳標準 LangChain tools。
開發者可以用原生 langchain-mcp-adapters 取代。

Usage:
    from mask.mcp.helpers import load_mcp_tools_from_config

    tools = load_mcp_tools_from_config(Path("config/mcp_servers.json"))
"""
```

從現有 `src/mask/mcp/` 提取 `load_mcp_tools_from_config` 函數，確保：
- 回傳 `list[BaseTool]`
- 支援 streamable-http 類型
- 處理檔案不存在的情況（回傳空 list）

### 1.2 建立 A2A Helpers

**新增檔案**: `src/mask/a2a/helpers.py`

```python
"""A2A Helper Functions.

這些是 helper functions，回傳原生 A2A SDK 類型。
開發者可以直接用 a2a-sdk 取代。

Usage:
    from mask.a2a.helpers import create_a2a_executor
    from a2a import A2AServer

    executor = create_a2a_executor(agent)
    server = A2AServer(executor)
"""
```

從現有 `src/mask/a2a/executor.py` 提取核心邏輯成 `create_a2a_executor` 函數。

### 1.3 更新 __init__.py

更新 `src/mask/mcp/__init__.py` 和 `src/mask/a2a/__init__.py` 的 exports。

### 1.4 標記 Legacy

在 `MaskA2AServer` 和 `SimpleAgent`/`BaseAgent` 加入 deprecation warning：
```python
import warnings
warnings.warn(
    "MaskA2AServer is deprecated. Use mask.a2a.helpers.create_a2a_executor instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

---

## Phase 2：CLI 模板文件化

### 2.1 建立模板目錄

建立 `src/mask/cli/templates/default/` 目錄結構：

```
src/mask/cli/templates/default/
├── pyproject.toml.jinja
├── README.md.jinja
├── .env.example.jinja
├── src/
│   └── {{ module_name }}/
│       ├── __init__.py.jinja
│       ├── agent.py.jinja
│       ├── main.py.jinja
│       ├── prompts/
│       │   └── system.md.jinja
│       ├── skills/
│       │   └── README.md.jinja
│       ├── tools/
│       │   ├── __init__.py.jinja
│       │   └── example.py.jinja
│       └── config/
│           └── mcp_servers.json.jinja
├── examples/
│   ├── skills/
│   │   └── pdf-processing/
│   │       └── SKILL.md.jinja
│   ├── tools/
│   │   └── weather.py.jinja
│   └── config/
│       └── mcp_servers.json.jinja
└── tests/
    ├── __init__.py.jinja
    └── test_agent.py.jinja
```

### 2.2 模板內容

#### agent.py.jinja
```python
"""Agent 設定 - 使用 LangChain create_agent + MASK SkillMiddleware"""
from pathlib import Path

from langchain.agents import create_agent

from mask.core import SkillRegistry
from mask.middleware import SkillMiddleware
from mask.models import LLMFactory, ModelTier
from mask.mcp.helpers import load_mcp_tools_from_config

from {{ module_name }}.tools import get_custom_tools


def load_system_prompt() -> str:
    prompt_file = Path(__file__).parent / "prompts" / "system.md"
    return prompt_file.read_text(encoding="utf-8")


def create_{{ module_name }}_agent(tier: ModelTier = ModelTier.THINKING):
    """建立 Agent

    使用：
    - LangChain create_agent (原生 API)
    - MASK SkillMiddleware (Progressive Disclosure)
    - 自定義 @tool
    - MCP tools (streamable-http)
    """
    model = LLMFactory().get_model(tier=tier)

    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    if skills_dir.exists():
        registry.discover_from_directory(skills_dir)

    mcp_config = Path(__file__).parent / "config" / "mcp_servers.json"

    tools = [
        *registry.get_all_tools(),
        *get_custom_tools(),
        *load_mcp_tools_from_config(mcp_config),
    ]

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=load_system_prompt(),
        middleware=[
            SkillMiddleware(registry),
        ],
    )
```

#### main.py.jinja
```python
"""A2A Server - 使用 mask.a2a helpers"""
import os
from pathlib import Path

from dotenv import load_dotenv
from a2a import A2AServer

from mask.a2a.helpers import create_a2a_executor
from mask.observability import setup_openinference_tracing

from {{ module_name }}.agent import create_{{ module_name }}_agent

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)


def main():
    setup_openinference_tracing(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME", "{{ project_name }}"),
        endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"),
        api_key=os.environ.get("PHOENIX_API_KEY"),
    )

    agent = create_{{ module_name }}_agent()
    executor = create_a2a_executor(agent)
    server = A2AServer(executor)

    print("Starting {{ project_name }} on port 10001...")
    server.run(port=10001)


if __name__ == "__main__":
    main()
```

#### tools/__init__.py.jinja
```python
"""自定義 Tools"""
from {{ module_name }}.tools.example import hello_world


def get_custom_tools() -> list:
    """取得所有自定義 tools"""
    return [
        hello_world,
    ]
```

#### tools/example.py.jinja
```python
"""範例 Tool - 使用 LangChain @tool decorator"""
from langchain_core.tools import tool


@tool
def hello_world(name: str) -> str:
    """Say hello to someone.

    Args:
        name: The name to greet.
    """
    return f"Hello, {name}!"
```

#### config/mcp_servers.json.jinja
```json
{
  "mcpServers": {
    "example-mcp-server": {
      "type": "streamable-http",
      "url": "http://localhost:9000/mcp/"
    }
  }
}
```

#### .env.example.jinja
```bash
# LLM Provider API Keys
ANTHROPIC_API_KEY=your-anthropic-key
# OPENAI_API_KEY=your-openai-key
# GOOGLE_API_KEY=your-google-key

# MASK Configuration
MASK_LLM_PROVIDER=anthropic

# Phoenix Observability
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME={{ project_name }}
# PHOENIX_API_KEY=your-phoenix-api-key

# MCP Server (example)
# EXAMPLE_API_KEY=your-api-key
```

#### tests/test_agent.py.jinja
```python
"""Agent 測試 - 驗證 API endpoint 和回覆"""
import pytest
import httpx


@pytest.fixture
def base_url():
    return "http://localhost:10001"


@pytest.mark.asyncio
async def test_health_endpoint(base_url):
    """測試 health endpoint 可存取"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/health")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_agent_card_endpoint(base_url):
    """測試 agent card endpoint"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/.well-known/agent.json")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data


@pytest.mark.asyncio
async def test_agent_can_respond(base_url):
    """測試 agent 能回覆訊息"""
    from uuid import uuid4

    async with httpx.AsyncClient() as client:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello, are you there?"}],
                    "messageId": uuid4().hex,
                    "contextId": uuid4().hex,
                }
            }
        }

        response = await client.post(f"{base_url}/", json=payload)
        assert response.status_code == 200
```

### 2.3 建立模板引擎

**新增檔案**: `src/mask/cli/template_engine.py`

使用 Jinja2 實作模板引擎：
- 從 `templates/default/` 讀取模板
- 處理目錄名稱中的 `{{ module_name }}`
- 渲染所有 `.jinja` 檔案

### 2.4 重構 init 命令

**修改檔案**: `src/mask/cli/commands/init.py`

- 移除所有 hardcoded strings
- 使用 `TemplateEngine` 渲染模板
- 移除 `--with-mcp` 和 `--with-a2a` flags（預設都有）
- 保留 `--stateless` flag

### 2.5 更新 pyproject.toml

加入 package-data 設定：
```toml
[tool.setuptools.package-data]
mask = ["cli/templates/**/*"]
```

---

## Phase 3：更新 Observability

### 3.1 支援 Phoenix API Key

**修改檔案**: `src/mask/observability/setup.py`

更新 `setup_openinference_tracing` 函數：
- 支援 `api_key` 參數
- 支援從環境變數讀取 `PHOENIX_PROJECT_NAME`
- 如果有 api_key，加到 OTEL headers

```python
def setup_openinference_tracing(
    project_name: str = None,
    endpoint: str = None,
    api_key: str = None,
    filter_a2a_noise: bool = True,
) -> None:
    # 從環境變數讀取預設值
    project_name = project_name or os.environ.get("PHOENIX_PROJECT_NAME", "mask-agent")
    endpoint = endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    api_key = api_key or os.environ.get("PHOENIX_API_KEY")

    # 設定 headers
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # ... 其餘設定
```

---

## Phase 4：測試與文件

### 4.1 測試

確保以下測試通過：
- `pytest tests/` - 現有測試
- 手動測試 `mask init test-project` 生成正確結構
- 手動測試生成的專案可以 `pip install -e .` 並執行

### 4.2 更新 CLAUDE.md

更新文件說明新的 helpers 用法和專案結構。

---

## 檔案變更總覽

### 新增檔案
- `src/mask/mcp/helpers.py`
- `src/mask/a2a/helpers.py`
- `src/mask/cli/template_engine.py`
- `src/mask/cli/templates/default/**` (所有模板檔案)

### 修改檔案
- `src/mask/mcp/__init__.py`
- `src/mask/a2a/__init__.py`
- `src/mask/a2a/executor.py` (加入 deprecation warning)
- `src/mask/a2a/server.py` (加入 deprecation warning)
- `src/mask/agent/base_agent.py` (加入 deprecation warning)
- `src/mask/cli/commands/init.py`
- `src/mask/observability/setup.py`
- `pyproject.toml`
- `CLAUDE.md`

---

## 執行指令

請依序執行以下步驟：

1. 執行 Phase 1：建立 helpers 模組
2. 執行 Phase 2：CLI 模板文件化
3. 執行 Phase 3：更新 Observability
4. 執行 Phase 4：測試並更新文件
5. 完成後 commit 並 push 到 `claude/simplify-sdk-wrappers-mIJvz` 分支
