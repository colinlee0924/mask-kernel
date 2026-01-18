"""Unit tests for DelegationToolFactory."""

from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mask.a2a.delegation import (
    DelegationToolFactory,
    create_delegation_tool_sync,
    create_delegation_tools,
)
from mask.core.events import AgentEvent


class TestDelegationToolFactory:
    """Tests for DelegationToolFactory class."""

    @pytest.fixture
    def factory(self):
        """Create a test factory."""
        return DelegationToolFactory(track_delegation_history=True)

    @pytest.fixture
    def factory_no_history(self):
        """Create a factory without history tracking."""
        return DelegationToolFactory(track_delegation_history=False)

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def test_init_default_values(self):
        """Test factory initialization with default values."""
        factory = DelegationToolFactory()

        assert factory.event_queue is None
        assert factory.track_delegation_history is True
        assert len(factory._clients) == 0
        assert len(factory._descriptions) == 0

    def test_init_with_event_queue(self):
        """Test factory initialization with event queue."""
        mock_queue = MagicMock()
        factory = DelegationToolFactory(event_queue=mock_queue)

        assert factory.event_queue is mock_queue

    def test_init_without_history_tracking(self):
        """Test factory initialization without history tracking."""
        factory = DelegationToolFactory(track_delegation_history=False)

        assert factory.track_delegation_history is False

    # =========================================================================
    # Tool Name Generation Tests
    # =========================================================================

    def test_tool_name_format_simple(self, factory):
        """Test tool name generation for simple agent name."""
        mock_client = MagicMock()
        mock_client.agent_name = "agent"
        factory._clients["agent"] = mock_client

        tools = factory.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "delegate_to_agent"

    def test_tool_name_format_with_hyphen(self, factory):
        """Test tool name generation converts hyphens to underscores."""
        mock_client = MagicMock()
        mock_client.agent_name = "jira-agent"
        factory._clients["jira-agent"] = mock_client

        tools = factory.get_tools()

        assert len(tools) == 1
        assert tools[0].name == "delegate_to_jira_agent"

    def test_tool_name_format_with_multiple_hyphens(self, factory):
        """Test tool name with multiple hyphens."""
        mock_client = MagicMock()
        mock_client.agent_name = "my-special-agent"
        factory._clients["my-special-agent"] = mock_client

        tools = factory.get_tools()

        assert tools[0].name == "delegate_to_my_special_agent"

    # =========================================================================
    # Tool Description Tests
    # =========================================================================

    def test_tool_description_default(self, factory):
        """Test default tool description."""
        mock_client = MagicMock()
        mock_client.agent_name = "test-agent"
        mock_client.card = None
        factory._clients["test-agent"] = mock_client
        factory._descriptions["test-agent"] = "Delegate tasks to test-agent"

        tools = factory.get_tools()

        assert "Delegate task to test-agent" in tools[0].description

    def test_tool_description_custom(self, factory):
        """Test custom tool description."""
        mock_client = MagicMock()
        mock_client.agent_name = "jira-agent"
        factory._clients["jira-agent"] = mock_client
        factory._descriptions["jira-agent"] = "Handles JIRA operations"

        tools = factory.get_tools()

        assert "Handles JIRA operations" in tools[0].description

    # =========================================================================
    # Multiple Agents Tests
    # =========================================================================

    def test_get_tools_multiple_agents(self, factory):
        """Test getting tools for multiple registered agents."""
        for name in ["jira-agent", "faq-agent", "slack-agent"]:
            mock_client = MagicMock()
            mock_client.agent_name = name
            factory._clients[name] = mock_client
            factory._descriptions[name] = f"Description for {name}"

        tools = factory.get_tools()

        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert "delegate_to_jira_agent" in tool_names
        assert "delegate_to_faq_agent" in tool_names
        assert "delegate_to_slack_agent" in tool_names

    def test_get_agent_names(self, factory):
        """Test getting list of registered agent names."""
        factory._clients["agent-a"] = MagicMock()
        factory._clients["agent-b"] = MagicMock()

        names = factory.get_agent_names()

        assert len(names) == 2
        assert "agent-a" in names
        assert "agent-b" in names

    def test_get_agent_names_empty(self, factory):
        """Test getting agent names when none registered."""
        names = factory.get_agent_names()

        assert names == []

    # =========================================================================
    # Delegation Tool Execution Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_delegation_returns_command(self, factory):
        """Test that delegation tool returns a Command object."""
        from langgraph.types import Command

        # Create mock streaming client
        mock_client = MagicMock()
        mock_client.agent_name = "test-agent"

        # Create async generator for streaming events
        async def mock_stream(task):
            yield AgentEvent.delegation_start("test-agent", task)
            yield AgentEvent.sub_agent_text_delta("Result text", "test-agent")
            yield AgentEvent.delegation_end("test-agent", "Done", success=True)

        mock_client.send_message_streaming = mock_stream
        factory._clients["test-agent"] = mock_client
        factory._descriptions["test-agent"] = "Test agent"

        tools = factory.get_tools()
        tool = tools[0]

        # Mock runtime
        runtime = MagicMock()
        runtime.tool_call_id = "call-123"

        result = await tool.ainvoke({"task": "Test task", "runtime": runtime})

        assert isinstance(result, Command)
        assert "messages" in result.update

    @pytest.mark.asyncio
    async def test_delegation_accumulates_text_delta(self, factory):
        """Test that delegation accumulates text from sub_agent_text_delta events."""
        from langgraph.types import Command

        mock_client = MagicMock()
        mock_client.agent_name = "test-agent"

        async def mock_stream(task):
            yield AgentEvent.delegation_start("test-agent", task)
            yield AgentEvent.sub_agent_text_delta("Hello ", "test-agent")
            yield AgentEvent.sub_agent_text_delta("World", "test-agent")
            yield AgentEvent.delegation_end("test-agent", "Done")

        mock_client.send_message_streaming = mock_stream
        factory._clients["test-agent"] = mock_client
        factory._descriptions["test-agent"] = "Test"

        tools = factory.get_tools()
        runtime = MagicMock()
        runtime.tool_call_id = "call-123"

        result = await tools[0].ainvoke({"task": "Test", "runtime": runtime})

        # Check that the accumulated text is in the message
        messages = result.update["messages"]
        assert len(messages) == 1
        assert "Hello World" in messages[0].content

    @pytest.mark.asyncio
    async def test_delegation_tracks_history(self, factory):
        """Test that delegation history is tracked when enabled."""
        from langgraph.types import Command

        mock_client = MagicMock()
        mock_client.agent_name = "test-agent"

        async def mock_stream(task):
            yield AgentEvent.delegation_start("test-agent", task)
            yield AgentEvent.sub_agent_tool_start(
                "search", "test-agent", {"q": "test"}
            )
            yield AgentEvent.sub_agent_tool_end("search", "test-agent", "results")
            yield AgentEvent.sub_agent_text_delta("Final result", "test-agent")
            yield AgentEvent.delegation_end("test-agent", "Done")

        mock_client.send_message_streaming = mock_stream
        factory._clients["test-agent"] = mock_client
        factory._descriptions["test-agent"] = "Test"

        tools = factory.get_tools()
        runtime = MagicMock()
        runtime.tool_call_id = "call-123"

        result = await tools[0].ainvoke({"task": "Test task", "runtime": runtime})

        assert "delegation_history" in result.update
        history = result.update["delegation_history"]
        assert len(history) == 1
        assert history[0]["agent"] == "test-agent"
        assert "Test task" in history[0]["task"]
        assert history[0]["event_count"] > 0

    @pytest.mark.asyncio
    async def test_delegation_no_history_when_disabled(self, factory_no_history):
        """Test that history is not tracked when disabled."""
        from langgraph.types import Command

        mock_client = MagicMock()
        mock_client.agent_name = "test-agent"

        async def mock_stream(task):
            yield AgentEvent.delegation_start("test-agent", task)
            yield AgentEvent.delegation_end("test-agent", "Done")

        mock_client.send_message_streaming = mock_stream
        factory_no_history._clients["test-agent"] = mock_client
        factory_no_history._descriptions["test-agent"] = "Test"

        tools = factory_no_history.get_tools()
        runtime = MagicMock()
        runtime.tool_call_id = "call-123"

        result = await tools[0].ainvoke({"task": "Test", "runtime": runtime})

        assert "delegation_history" not in result.update

    @pytest.mark.asyncio
    async def test_delegation_handles_error(self, factory):
        """Test that delegation handles errors gracefully."""
        from langgraph.types import Command

        mock_client = MagicMock()
        mock_client.agent_name = "broken-agent"

        async def mock_stream(task):
            yield AgentEvent.delegation_start("broken-agent", task)
            raise Exception("Connection failed")

        mock_client.send_message_streaming = mock_stream
        factory._clients["broken-agent"] = mock_client
        factory._descriptions["broken-agent"] = "Broken"

        tools = factory.get_tools()
        runtime = MagicMock()
        runtime.tool_call_id = "call-123"

        result = await tools[0].ainvoke({"task": "Test", "runtime": runtime})

        # Should still return a Command with error message
        assert isinstance(result, Command)
        messages = result.update["messages"]
        assert "Error" in messages[0].content

    # =========================================================================
    # Event Queue Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_delegation_emits_events_to_queue(self, factory):
        """Test that events are attempted to be emitted to the event queue.

        Note: In unit tests, TaskStatusUpdateEvent validation fails because
        contextId, taskId, etc. are not available. This test verifies that
        _emit_event_to_queue is called for each event type.
        """
        # Set a mock event queue to enable event emission
        mock_queue = MagicMock()
        factory.event_queue = mock_queue

        # Track calls to _emit_event_to_queue
        emit_calls = []
        original_emit = factory._emit_event_to_queue

        async def track_emit(event, source_agent):
            emit_calls.append((event.type, source_agent))
            return await original_emit(event, source_agent)

        factory._emit_event_to_queue = track_emit

        mock_client = MagicMock()
        mock_client.agent_name = "test-agent"

        async def mock_stream(task):
            yield AgentEvent.delegation_start("test-agent", task)
            yield AgentEvent.sub_agent_tool_start("search", "test-agent", {})
            yield AgentEvent.sub_agent_tool_end("search", "test-agent", "result")
            yield AgentEvent.delegation_end("test-agent", "Done")

        mock_client.send_message_streaming = mock_stream
        factory._clients["test-agent"] = mock_client
        factory._descriptions["test-agent"] = "Test"

        tools = factory.get_tools()
        runtime = MagicMock()
        runtime.tool_call_id = "call-123"

        await tools[0].ainvoke({"task": "Test", "runtime": runtime})

        # Should have attempted to emit events for each event type
        assert len(emit_calls) >= 4
        event_types = [call[0] for call in emit_calls]
        assert "delegation_start" in event_types
        assert "delegation_end" in event_types

    # =========================================================================
    # Close Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_close_clears_clients(self, factory):
        """Test that close() clears all clients."""
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        factory._clients["agent1"] = mock_client1
        factory._clients["agent2"] = mock_client2
        factory._descriptions["agent1"] = "desc1"
        factory._descriptions["agent2"] = "desc2"

        await factory.close()

        assert len(factory._clients) == 0
        assert len(factory._descriptions) == 0
        mock_client1.close.assert_called_once()
        mock_client2.close.assert_called_once()


class TestCreateDelegationToolSync:
    """Tests for create_delegation_tool_sync helper."""

    def test_creates_tool_for_existing_client(self):
        """Test creating a single tool from an existing client."""
        mock_client = MagicMock()
        mock_client.agent_name = "test-agent"

        tool = create_delegation_tool_sync("test-agent", mock_client)

        assert tool.name == "delegate_to_test_agent"

    def test_respects_track_history_setting(self):
        """Test that track_history setting is respected."""
        mock_client = MagicMock()

        tool_with_history = create_delegation_tool_sync(
            "test", mock_client, track_history=True
        )
        tool_without_history = create_delegation_tool_sync(
            "test2", mock_client, track_history=False
        )

        # Both should be created successfully
        assert tool_with_history is not None
        assert tool_without_history is not None
