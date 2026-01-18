"""Integration tests for delegation with mock A2A server."""

import pytest
from httpx import ASGITransport, AsyncClient

from mask.a2a.streaming_client import StreamingA2AClient
from mask.core.events import AgentEvent


class TestStreamingClientIntegration:
    """Integration tests for StreamingA2AClient with mock server."""

    @pytest.fixture
    def mock_server(self):
        """Get the mock A2A server app."""
        from tests.integration.mock_a2a_server import app

        return app

    @pytest.mark.asyncio
    async def test_connect_discovers_agent_name(self, mock_server):
        """Test that agent card endpoint returns valid card data."""
        transport = ASGITransport(app=mock_server)

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            # Directly fetch the agent card endpoint
            response = await http.get("/.well-known/agent.json")

            assert response.status_code == 200
            card_data = response.json()

            assert card_data["name"] == "mock-agent"
            assert card_data["description"] == "A mock A2A agent for testing"
            assert "capabilities" in card_data
            assert "skills" in card_data

    @pytest.mark.asyncio
    async def test_streaming_yields_delegation_events(self, mock_server):
        """Test that streaming yields delegation_start and delegation_end."""
        transport = ASGITransport(app=mock_server)

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            client = StreamingA2AClient("http://test", agent_name="mock-agent")
            client._http_client = http

            events = []
            async for event in client.send_message_streaming("Test message"):
                events.append(event)

            # Should have delegation_start at beginning
            assert events[0].type == "delegation_start"
            assert events[0].name == "mock-agent"

            # Should have delegation_end at end
            assert events[-1].type == "delegation_end"
            assert events[-1].name == "mock-agent"

    @pytest.mark.asyncio
    async def test_streaming_yields_tool_events(self, mock_server):
        """Test that streaming yields sub-agent tool events."""
        transport = ASGITransport(app=mock_server)

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            client = StreamingA2AClient("http://test", agent_name="mock-agent")
            client._http_client = http

            events = []
            async for event in client.send_message_streaming("Test"):
                events.append(event)

            # Find tool events
            tool_start_events = [e for e in events if e.type == "sub_agent_tool_start"]
            tool_end_events = [e for e in events if e.type == "sub_agent_tool_end"]

            assert len(tool_start_events) >= 1
            assert len(tool_end_events) >= 1

            # Check tool name
            assert tool_start_events[0].name == "mock_search"
            assert tool_end_events[0].name == "mock_search"

    @pytest.mark.asyncio
    async def test_streaming_yields_text_delta(self, mock_server):
        """Test that streaming yields sub_agent_text_delta events."""
        transport = ASGITransport(app=mock_server)

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            client = StreamingA2AClient("http://test", agent_name="mock-agent")
            client._http_client = http

            events = []
            async for event in client.send_message_streaming("Test"):
                events.append(event)

            # Find text delta events
            text_events = [e for e in events if e.type == "sub_agent_text_delta"]

            assert len(text_events) >= 1
            assert text_events[0].source_agent == "mock-agent"

    @pytest.mark.asyncio
    async def test_streaming_accumulates_final_result(self, mock_server):
        """Test that delegation_end contains accumulated result."""
        transport = ASGITransport(app=mock_server)

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            client = StreamingA2AClient("http://test", agent_name="mock-agent")
            client._http_client = http

            events = []
            async for event in client.send_message_streaming("Test"):
                events.append(event)

            # Check delegation_end result
            end_event = events[-1]
            assert end_event.type == "delegation_end"
            # Result should contain accumulated text
            assert len(end_event.data.get("result", "")) > 0

    @pytest.mark.asyncio
    async def test_streaming_sets_source_agent(self, mock_server):
        """Test that all sub-agent events have source_agent set."""
        transport = ASGITransport(app=mock_server)

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            client = StreamingA2AClient("http://test", agent_name="test-source")
            client._http_client = http

            events = []
            async for event in client.send_message_streaming("Test"):
                events.append(event)

            # All sub_agent_* events should have source_agent
            sub_agent_events = [e for e in events if e.type.startswith("sub_agent_")]
            for event in sub_agent_events:
                assert event.source_agent == "test-source"


class TestDelegationFactoryIntegration:
    """Integration tests for DelegationToolFactory with mock server."""

    @pytest.fixture
    def mock_server(self):
        """Get the mock A2A server app."""
        from tests.integration.mock_a2a_server import app

        return app

    @pytest.mark.asyncio
    async def test_delegation_tool_returns_command(self, mock_server):
        """Test that delegation tool returns a Command with results."""
        from unittest.mock import MagicMock

        from langgraph.types import Command

        from mask.a2a.delegation import DelegationToolFactory

        transport = ASGITransport(app=mock_server)

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            # Create a streaming client manually
            client = StreamingA2AClient("http://test", agent_name="mock-agent")
            client._http_client = http

            # Create factory and add client
            factory = DelegationToolFactory(track_delegation_history=True)
            factory._clients["mock-agent"] = client
            factory._descriptions["mock-agent"] = "Mock agent for testing"

            tools = factory.get_tools()
            assert len(tools) == 1

            tool = tools[0]
            assert tool.name == "delegate_to_mock_agent"

            # Execute tool
            runtime = MagicMock()
            runtime.tool_call_id = "call-123"

            result = await tool.ainvoke({"task": "Test delegation", "runtime": runtime})

            # Check result
            assert isinstance(result, Command)
            assert "messages" in result.update
            assert len(result.update["messages"]) == 1

            # Check message content
            msg = result.update["messages"][0]
            assert "[mock-agent]" in msg.content

    @pytest.mark.asyncio
    async def test_delegation_tool_tracks_history(self, mock_server):
        """Test that delegation history is tracked."""
        from unittest.mock import MagicMock

        from mask.a2a.delegation import DelegationToolFactory

        transport = ASGITransport(app=mock_server)

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            client = StreamingA2AClient("http://test", agent_name="mock-agent")
            client._http_client = http

            factory = DelegationToolFactory(track_delegation_history=True)
            factory._clients["mock-agent"] = client
            factory._descriptions["mock-agent"] = "Mock"

            tools = factory.get_tools()
            runtime = MagicMock()
            runtime.tool_call_id = "call-123"

            result = await tools[0].ainvoke(
                {"task": "Track this task", "runtime": runtime}
            )

            # Check history
            assert "delegation_history" in result.update
            history = result.update["delegation_history"]
            assert len(history) == 1
            assert history[0]["agent"] == "mock-agent"
            assert "Track this task" in history[0]["task"]
            assert history[0]["event_count"] > 0

    @pytest.mark.asyncio
    async def test_delegation_tool_emits_to_queue(self, mock_server):
        """Test that delegation attempts to emit events to the queue.

        Note: In tests, TaskStatusUpdateEvent validation fails because
        contextId, taskId, etc. are not available. We verify by tracking
        calls to the internal _emit_event_to_queue method.
        """
        from unittest.mock import MagicMock, patch

        from mask.a2a.delegation import DelegationToolFactory

        transport = ASGITransport(app=mock_server)

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            client = StreamingA2AClient("http://test", agent_name="mock-agent")
            client._http_client = http

            mock_queue = MagicMock()
            factory = DelegationToolFactory(
                event_queue=mock_queue, track_delegation_history=False
            )
            factory._clients["mock-agent"] = client
            factory._descriptions["mock-agent"] = "Mock"

            # Track calls to _emit_event_to_queue
            emit_calls = []
            original_emit = factory._emit_event_to_queue

            async def track_emit(event, source_agent):
                emit_calls.append((event.type, source_agent))
                return await original_emit(event, source_agent)

            factory._emit_event_to_queue = track_emit

            tools = factory.get_tools()
            runtime = MagicMock()
            runtime.tool_call_id = "call-123"

            await tools[0].ainvoke({"task": "Test", "runtime": runtime})

            # Should have attempted to emit events (even if TaskStatusUpdateEvent fails)
            assert len(emit_calls) > 0
            event_types = [call[0] for call in emit_calls]
            # Should at least have delegation_start
            assert "delegation_start" in event_types or "sub_agent_tool_start" in event_types


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    @pytest.fixture
    def error_server(self):
        """Get a mock server that simulates errors."""
        from tests.integration.mock_a2a_server import create_mock_app_with_error

        return create_mock_app_with_error()

    @pytest.mark.asyncio
    async def test_streaming_handles_server_error(self, error_server):
        """Test that streaming handles server errors gracefully."""
        transport = ASGITransport(app=error_server)

        async with AsyncClient(transport=transport, base_url="http://test") as http:
            client = StreamingA2AClient("http://test", agent_name="error-agent")
            client._http_client = http

            events = []
            async for event in client.send_message_streaming("Test"):
                events.append(event)

            # Should still have delegation_start and delegation_end
            assert events[0].type == "delegation_start"
            assert events[-1].type == "delegation_end"

    @pytest.mark.asyncio
    async def test_delegation_handles_connection_error(self):
        """Test that delegation handles connection errors."""
        from unittest.mock import MagicMock

        from mask.a2a.delegation import DelegationToolFactory

        # Create client pointing to non-existent server
        client = StreamingA2AClient("http://nonexistent:99999", agent_name="bad-agent")

        factory = DelegationToolFactory()
        factory._clients["bad-agent"] = client
        factory._descriptions["bad-agent"] = "Bad"

        tools = factory.get_tools()
        runtime = MagicMock()
        runtime.tool_call_id = "call-123"

        # Should not raise, should return error in Command
        result = await tools[0].ainvoke({"task": "Test", "runtime": runtime})

        assert "Error" in result.update["messages"][0].content
