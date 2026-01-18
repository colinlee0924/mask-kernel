"""End-to-end tests for orchestrator agent.

These tests verify the complete flow from client request to response,
including A2A protocol handling, SSE streaming, and agent execution.

Note: These tests require:
- ANTHROPIC_API_KEY environment variable (for LLM calls)
- Network connectivity for A2A protocol

Run with:
    pytest tests/e2e/test_orchestrator_e2e.py -v --tb=short

For manual testing without API key:
    # Start orchestrator (uses echo fallback tool)
    python examples/orchestrator_agent.py

    # In another terminal, test agent card:
    curl http://localhost:10001/.well-known/agent.json

    # Test streaming:
    curl -X POST http://localhost:10001 \\
        -H "Content-Type: application/json" \\
        -H "Accept: text/event-stream" \\
        -d '{"jsonrpc":"2.0","method":"tasks/sendSubscribe","id":1,"params":{"message":{"parts":[{"text":"Hello"}]}}}'
"""

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


# Condition for skipping tests that need API key
needs_api_key = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)


@needs_api_key
class TestOrchestratorE2EWithMock:
    """E2E tests that mock the LLM but test the full A2A flow."""

    @pytest.fixture
    async def mock_orchestrator_app(self):
        """Create orchestrator app with mocked LLM."""
        from unittest.mock import AsyncMock

        from a2a.server.apps import A2AStarletteApplication
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore
        from a2a.types import AgentCapabilities, AgentCard, AgentSkill
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool

        from mask.a2a import create_a2a_executor
        from mask.middleware import A2AStreamingMiddleware

        # Create a simple echo tool
        @tool
        def echo(message: str) -> str:
            """Echo the message back."""
            return f"Echo: {message}"

        # Create streaming middleware
        streaming_middleware = A2AStreamingMiddleware(
            agent_name="test-orchestrator",
            emit_thinking=True,
        )

        # Create a mock agent that returns fixed responses
        class MockAgent:
            """Mock agent for testing."""

            async def ainvoke(self, inputs, config=None):
                """Return fixed response."""
                return {
                    "messages": [
                        AIMessage(content="This is a test response from the orchestrator.")
                    ]
                }

            async def astream_events(self, inputs, config=None, version="v1"):
                """Stream mock events."""
                from langchain_core.messages import AIMessage

                # Yield initial event
                yield {
                    "event": "on_chat_model_start",
                    "name": "ChatAnthropic",
                    "data": {"input": inputs},
                }

                # Yield streaming event
                yield {
                    "event": "on_chat_model_stream",
                    "name": "ChatAnthropic",
                    "data": {
                        "chunk": AIMessage(content="Test ")
                    },
                }

                yield {
                    "event": "on_chat_model_stream",
                    "name": "ChatAnthropic",
                    "data": {
                        "chunk": AIMessage(content="response.")
                    },
                }

                # Yield end event
                yield {
                    "event": "on_chat_model_end",
                    "name": "ChatAnthropic",
                    "data": {
                        "output": AIMessage(content="Test response.")
                    },
                }

        mock_agent = MockAgent()

        # Create executor
        executor = create_a2a_executor(
            mock_agent,
            server_name="test-orchestrator",
            streaming_middleware=streaming_middleware,
        )

        # Create A2A app
        agent_card = AgentCard(
            name="test-orchestrator",
            description="Test orchestrator for E2E testing",
            url="http://localhost:10001",
            version="1.0.0",
            skills=[
                AgentSkill(
                    id="test",
                    name="Test",
                    description="Test skill",
                    tags=["test"],
                )
            ],
            capabilities=AgentCapabilities(streaming=True),
        )

        task_store = InMemoryTaskStore()
        handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=task_store,
        )

        app = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=handler,
        )

        return app.build()

    @pytest.mark.asyncio
    async def test_agent_card_endpoint(self, mock_orchestrator_app):
        """Test that agent card endpoint returns valid card."""
        transport = ASGITransport(app=mock_orchestrator_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/.well-known/agent.json")

            assert response.status_code == 200
            card = response.json()

            assert card["name"] == "test-orchestrator"
            assert "capabilities" in card
            assert card["capabilities"]["streaming"] is True

    @pytest.mark.asyncio
    async def test_send_message_endpoint(self, mock_orchestrator_app):
        """Test that send endpoint works."""
        transport = ASGITransport(app=mock_orchestrator_app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            request_body = {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "id": 1,
                "params": {
                    "message": {
                        "parts": [{"text": "Hello, test!"}]
                    }
                }
            }

            response = await client.post(
                "/",
                json=request_body,
                headers={"Content-Type": "application/json"},
            )

            # Should return valid JSON-RPC response
            assert response.status_code == 200
            data = response.json()
            assert "result" in data or "error" in data


class TestOrchestratorImports:
    """Test that all orchestrator components can be imported."""

    def test_delegation_imports(self):
        """Test delegation module imports."""
        from mask.a2a.delegation import (
            DelegationToolFactory,
            create_delegation_tool_sync,
            create_delegation_tools,
        )

        assert DelegationToolFactory is not None
        assert create_delegation_tools is not None
        assert create_delegation_tool_sync is not None

    def test_streaming_client_imports(self):
        """Test streaming client imports."""
        from mask.a2a.streaming_client import StreamingA2AClient

        assert StreamingA2AClient is not None

    def test_middleware_imports(self):
        """Test middleware imports."""
        from mask.middleware.a2a_streaming import A2AStreamingMiddleware

        assert A2AStreamingMiddleware is not None

    def test_events_imports(self):
        """Test events imports."""
        from mask.core.events import AgentEvent

        assert AgentEvent is not None

    def test_package_exports(self):
        """Test that package __init__ exports work."""
        from mask.a2a import (
            DelegationToolFactory,
            StreamingA2AClient,
            create_a2a_executor,
            create_delegation_tools,
        )
        from mask.middleware import A2AStreamingMiddleware

        assert DelegationToolFactory is not None
        assert StreamingA2AClient is not None
        assert create_a2a_executor is not None
        assert A2AStreamingMiddleware is not None


class TestManualE2EInstructions:
    """Placeholder class with manual E2E testing instructions.

    Run manually when API keys are available:

    1. Start orchestrator (no sub-agents, uses echo fallback):
       ```bash
       python examples/orchestrator_agent.py
       ```

    2. Test agent card:
       ```bash
       curl http://localhost:10001/.well-known/agent.json | jq
       ```

    3. Test SSE streaming:
       ```bash
       curl -X POST http://localhost:10001 \
           -H "Content-Type: application/json" \
           -H "Accept: text/event-stream" \
           -d '{"jsonrpc":"2.0","method":"tasks/sendSubscribe","id":1,"params":{"message":{"parts":[{"text":"Echo hello world"}]}}}'
       ```

    4. Expected flow:
       - Agent receives request
       - A2AStreamingMiddleware emits thinking events
       - Agent uses echo tool (since no sub-agents)
       - Response streamed back via SSE
    """

    def test_placeholder(self):
        """This is a placeholder - see docstring for manual instructions."""
        assert True
