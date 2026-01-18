"""Unit tests for StreamingA2AClient."""

import json

import pytest

from mask.a2a.streaming_client import StreamingA2AClient


class TestStreamingA2AClient:
    """Tests for StreamingA2AClient class."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return StreamingA2AClient(
            "http://localhost:10001",
            agent_name="test-agent",
            timeout=30.0,
        )

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def test_init_with_trailing_slash(self):
        """Test that trailing slash is removed from base_url."""
        client = StreamingA2AClient("http://localhost:10001/")
        assert client.base_url == "http://localhost:10001"

    def test_init_without_trailing_slash(self):
        """Test base_url without trailing slash."""
        client = StreamingA2AClient("http://localhost:10001")
        assert client.base_url == "http://localhost:10001"

    def test_init_with_agent_name(self):
        """Test initialization with explicit agent name."""
        client = StreamingA2AClient(
            "http://localhost:10001",
            agent_name="custom-agent",
        )
        assert client.agent_name == "custom-agent"

    def test_init_without_agent_name(self):
        """Test initialization without agent name (will be discovered)."""
        client = StreamingA2AClient("http://localhost:10001")
        assert client.agent_name is None

    def test_init_default_timeout(self):
        """Test default timeout value."""
        client = StreamingA2AClient("http://localhost:10001")
        assert client.timeout == 120.0

    def test_init_custom_timeout(self):
        """Test custom timeout value."""
        client = StreamingA2AClient("http://localhost:10001", timeout=60.0)
        assert client.timeout == 60.0

    def test_initial_state(self, client):
        """Test initial client state."""
        assert client._card is None
        assert client._http_client is None

    # =========================================================================
    # SSE Parsing Tests
    # =========================================================================

    def test_parse_sse_event_with_data_prefix(self, client):
        """Test parsing SSE event with 'data:' prefix."""
        event_str = 'data: {"result": {"state": "working"}}'
        result = client._parse_sse_event(event_str)

        assert result is not None
        assert "result" in result
        assert result["result"]["state"] == "working"

    def test_parse_sse_event_with_whitespace(self, client):
        """Test parsing SSE event with extra whitespace."""
        event_str = '  data:   {"key": "value"}  '
        result = client._parse_sse_event(event_str)

        assert result is not None
        assert result["key"] == "value"

    def test_parse_sse_event_multiline(self, client):
        """Test parsing multiline SSE event."""
        event_str = 'id: 123\nevent: message\ndata: {"test": true}'
        result = client._parse_sse_event(event_str)

        assert result is not None
        assert result["test"] is True

    def test_parse_sse_event_no_data(self, client):
        """Test parsing SSE event without data line."""
        event_str = "id: 123\nevent: ping"
        result = client._parse_sse_event(event_str)

        assert result is None

    def test_parse_sse_event_invalid_json(self, client):
        """Test parsing SSE event with invalid JSON."""
        event_str = "data: {invalid json}"
        result = client._parse_sse_event(event_str)

        assert result is None

    def test_parse_sse_event_empty_data(self, client):
        """Test parsing SSE event with empty data."""
        event_str = "data:"
        result = client._parse_sse_event(event_str)

        assert result is None

    # =========================================================================
    # Event Conversion Tests - TaskStatusUpdateEvent
    # =========================================================================

    def test_convert_tool_start_status_event(self, client):
        """Test converting tool_start status event to AgentEvent."""
        client.agent_name = "jira-agent"
        sse_data = {
            "result": {
                "state": "working",
                "message": {
                    "parts": [
                        {"text": "Running tool"},
                        {
                            "data": {
                                "event_type": "tool_start",
                                "tool_name": "jira_search",
                                "input": {"query": "bugs"},
                            }
                        },
                    ]
                },
            }
        }

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is not None
        assert event.type == "sub_agent_tool_start"
        assert event.name == "jira_search"
        assert event.source_agent == "jira-agent"
        assert event.data["input"]["query"] == "bugs"

    def test_convert_tool_end_status_event(self, client):
        """Test converting tool_end status event to AgentEvent."""
        client.agent_name = "jira-agent"
        sse_data = {
            "result": {
                "state": "working",
                "message": {
                    "parts": [
                        {
                            "data": {
                                "event_type": "tool_end",
                                "tool_name": "jira_search",
                                "output": "Found 5 issues",
                                "duration_ms": 1500,
                            }
                        }
                    ]
                },
            }
        }

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is not None
        assert event.type == "sub_agent_tool_end"
        assert event.data["output"] == "Found 5 issues"
        assert event.data["duration_ms"] == 1500

    def test_convert_thinking_status_event(self, client):
        """Test converting llm_thinking status event to AgentEvent."""
        client.agent_name = "analyzer"
        sse_data = {
            "result": {
                "state": "working",
                "message": {
                    "parts": [{"data": {"event_type": "llm_thinking"}}]
                },
            }
        }

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is not None
        assert event.type == "sub_agent_thinking"
        assert event.source_agent == "analyzer"

    # =========================================================================
    # Event Conversion Tests - TaskArtifactUpdateEvent
    # =========================================================================

    def test_convert_response_artifact(self, client):
        """Test converting response artifact to text delta event."""
        client.agent_name = "faq-agent"
        sse_data = {
            "result": {
                "artifact": {
                    "name": "response",
                    "parts": [{"text": "Here is the answer..."}],
                }
            }
        }

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is not None
        assert event.type == "sub_agent_text_delta"
        assert event.data["delta"] == "Here is the answer..."
        assert event.source_agent == "faq-agent"

    def test_convert_tool_call_artifact(self, client):
        """Test converting tool_call artifact to sub_agent_tool_start."""
        client.agent_name = "worker"
        sse_data = {
            "result": {
                "artifact": {
                    "name": "tool_call",
                    "parts": [
                        {
                            "text": json.dumps(
                                {"tool": "search", "input": {"query": "test"}}
                            )
                        }
                    ],
                }
            }
        }

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is not None
        assert event.type == "sub_agent_tool_start"
        assert event.name == "search"
        assert event.data["input"]["query"] == "test"

    def test_convert_tool_result_artifact(self, client):
        """Test converting tool_result artifact to sub_agent_tool_end."""
        client.agent_name = "worker"
        sse_data = {
            "result": {
                "artifact": {
                    "name": "tool_result",
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "tool": "search",
                                    "output": "Results here",
                                    "duration_ms": 500,
                                }
                            )
                        }
                    ],
                }
            }
        }

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is not None
        assert event.type == "sub_agent_tool_end"
        assert event.name == "search"
        assert event.data["output"] == "Results here"
        assert event.data["duration_ms"] == 500

    def test_convert_tool_result_plain_text(self, client):
        """Test converting plain text tool_result artifact."""
        client.agent_name = "worker"
        sse_data = {
            "result": {
                "artifact": {
                    "name": "tool_result",
                    "parts": [{"text": "Plain text result"}],
                }
            }
        }

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is not None
        assert event.type == "sub_agent_tool_end"
        assert event.data["output"] == "Plain text result"

    def test_convert_thinking_artifact(self, client):
        """Test converting thinking artifact to sub_agent_thinking."""
        client.agent_name = "thinker"
        sse_data = {
            "result": {
                "artifact": {
                    "name": "thinking",
                    "parts": [{"text": "Considering options..."}],
                }
            }
        }

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is not None
        assert event.type == "sub_agent_thinking"
        assert event.data["content"] == "Considering options..."

    def test_convert_unknown_artifact(self, client):
        """Test that unknown artifacts return None."""
        client.agent_name = "unknown"
        sse_data = {
            "result": {
                "artifact": {
                    "name": "unknown_type",
                    "parts": [{"text": "some data"}],
                }
            }
        }

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is None

    def test_convert_empty_result(self, client):
        """Test converting event with no recognizable content."""
        client.agent_name = "test"
        sse_data = {"result": {"unknown_field": "value"}}

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is None

    def test_convert_handles_missing_agent_name(self, client):
        """Test conversion when agent_name is None."""
        client.agent_name = None
        sse_data = {
            "result": {
                "artifact": {
                    "name": "response",
                    "parts": [{"text": "response text"}],
                }
            }
        }

        event = client._convert_to_agent_event(sse_data, "run-1")

        assert event is not None
        assert event.source_agent == "unknown"

    # =========================================================================
    # Runtime Error Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_send_message_without_connect_raises(self, client):
        """Test that sending message without connect() raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Client not connected"):
            async for _ in client.send_message_streaming("test"):
                pass


class TestCreateStreamingClient:
    """Tests for create_streaming_client helper function."""

    # Note: Integration tests for actual connection are in integration tests
    pass
