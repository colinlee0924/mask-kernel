"""Unit tests for AgentEvent class."""

import pytest

from mask.core.events import AgentEvent


class TestAgentEvent:
    """Tests for AgentEvent dataclass and factory methods."""

    # =========================================================================
    # Basic Event Creation
    # =========================================================================

    def test_text_delta_factory(self):
        """Test creating a text delta event."""
        event = AgentEvent.text_delta("Hello world", run_id="run-123")

        assert event.type == "text_delta"
        assert event.data["delta"] == "Hello world"
        assert event.run_id == "run-123"
        assert event.source_agent == ""

    def test_tool_start_factory(self):
        """Test creating a tool start event."""
        event = AgentEvent.tool_start(
            name="jira_search",
            run_id="run-123",
            input_data={"query": "open bugs"},
        )

        assert event.type == "tool_call_start"
        assert event.name == "jira_search"
        assert event.run_id == "run-123"
        assert event.data["input"]["query"] == "open bugs"

    def test_tool_end_factory(self):
        """Test creating a tool end event."""
        event = AgentEvent.tool_end(
            name="jira_search",
            run_id="run-123",
            output="Found 5 issues",
        )

        assert event.type == "tool_call_end"
        assert event.name == "jira_search"
        assert event.data["output"] == "Found 5 issues"

    def test_agent_start_factory(self):
        """Test creating an agent start event."""
        event = AgentEvent.agent_start(name="orchestrator", run_id="run-123")

        assert event.type == "agent_start"
        assert event.name == "orchestrator"
        assert event.run_id == "run-123"

    def test_agent_end_factory(self):
        """Test creating an agent end event."""
        event = AgentEvent.agent_end(name="orchestrator", run_id="run-123")

        assert event.type == "agent_end"
        assert event.name == "orchestrator"

    def test_error_factory(self):
        """Test creating an error event."""
        event = AgentEvent.error("Connection failed", run_id="run-123")

        assert event.type == "error"
        assert event.data["message"] == "Connection failed"

    # =========================================================================
    # Delegation Events
    # =========================================================================

    def test_delegation_start_factory(self):
        """Test creating a delegation start event."""
        event = AgentEvent.delegation_start(
            target_agent="jira-agent",
            task="Search for open bugs",
            run_id="run-123",
        )

        assert event.type == "delegation_start"
        assert event.name == "jira-agent"
        assert event.data["task"] == "Search for open bugs"
        assert event.data["target_agent"] == "jira-agent"

    def test_delegation_end_factory_success(self):
        """Test creating a successful delegation end event."""
        event = AgentEvent.delegation_end(
            target_agent="jira-agent",
            result="Found 3 bugs",
            run_id="run-123",
            success=True,
        )

        assert event.type == "delegation_end"
        assert event.name == "jira-agent"
        assert event.data["result"] == "Found 3 bugs"
        assert event.data["success"] is True

    def test_delegation_end_factory_failure(self):
        """Test creating a failed delegation end event."""
        event = AgentEvent.delegation_end(
            target_agent="jira-agent",
            result="Connection timeout",
            run_id="run-123",
            success=False,
        )

        assert event.type == "delegation_end"
        assert event.data["success"] is False

    # =========================================================================
    # Sub-Agent Events
    # =========================================================================

    def test_from_sub_agent_tool_start(self):
        """Test converting tool_call_start to sub_agent_tool_start."""
        original = AgentEvent.tool_start(
            name="search",
            run_id="run-1",
            input_data={"query": "test"},
        )

        converted = AgentEvent.from_sub_agent(original, "jira-agent")

        assert converted.type == "sub_agent_tool_start"
        assert converted.source_agent == "jira-agent"
        assert converted.name == "search"
        assert converted.data["input"]["query"] == "test"

    def test_from_sub_agent_tool_end(self):
        """Test converting tool_call_end to sub_agent_tool_end."""
        original = AgentEvent.tool_end(
            name="search",
            run_id="run-1",
            output="results",
        )

        converted = AgentEvent.from_sub_agent(original, "faq-agent")

        assert converted.type == "sub_agent_tool_end"
        assert converted.source_agent == "faq-agent"

    def test_from_sub_agent_text_delta(self):
        """Test converting text_delta to sub_agent_text_delta."""
        original = AgentEvent.text_delta("Hello", run_id="run-1")

        converted = AgentEvent.from_sub_agent(original, "helper-agent")

        assert converted.type == "sub_agent_text_delta"
        assert converted.source_agent == "helper-agent"
        assert converted.data["delta"] == "Hello"

    def test_from_sub_agent_error(self):
        """Test converting error to sub_agent_error."""
        original = AgentEvent.error("Failed", run_id="run-1")

        converted = AgentEvent.from_sub_agent(original, "broken-agent")

        assert converted.type == "sub_agent_error"
        assert converted.source_agent == "broken-agent"

    def test_sub_agent_tool_start_factory(self):
        """Test creating sub-agent tool start event directly."""
        event = AgentEvent.sub_agent_tool_start(
            tool_name="jira_search",
            source_agent="jira-agent",
            input_data={"query": "AI bugs"},
            run_id="run-123",
        )

        assert event.type == "sub_agent_tool_start"
        assert event.name == "jira_search"
        assert event.source_agent == "jira-agent"
        assert event.data["input"]["query"] == "AI bugs"

    def test_sub_agent_tool_end_factory(self):
        """Test creating sub-agent tool end event directly."""
        event = AgentEvent.sub_agent_tool_end(
            tool_name="jira_search",
            source_agent="jira-agent",
            output="Found 3 issues",
            run_id="run-123",
            duration_ms=1500,
        )

        assert event.type == "sub_agent_tool_end"
        assert event.data["output"] == "Found 3 issues"
        assert event.data["duration_ms"] == 1500

    def test_sub_agent_text_delta_factory(self):
        """Test creating sub-agent text delta event directly."""
        event = AgentEvent.sub_agent_text_delta(
            content="Analysis complete",
            source_agent="analyzer-agent",
            run_id="run-123",
        )

        assert event.type == "sub_agent_text_delta"
        assert event.source_agent == "analyzer-agent"
        assert event.data["delta"] == "Analysis complete"

    def test_sub_agent_error_factory(self):
        """Test creating sub-agent error event directly."""
        event = AgentEvent.sub_agent_error(
            message="Timeout occurred",
            source_agent="slow-agent",
            run_id="run-123",
        )

        assert event.type == "sub_agent_error"
        assert event.source_agent == "slow-agent"
        assert event.data["message"] == "Timeout occurred"

    # =========================================================================
    # Serialization
    # =========================================================================

    def test_to_dict_basic(self):
        """Test converting event to dictionary."""
        event = AgentEvent.tool_start(
            name="search",
            run_id="run-123",
            input_data={"query": "test"},
        )

        d = event.to_dict()

        assert d["type"] == "tool_call_start"
        assert d["name"] == "search"
        assert d["run_id"] == "run-123"
        assert d["data"]["input"]["query"] == "test"
        assert "source_agent" not in d  # Empty source_agent not included

    def test_to_dict_includes_source_agent(self):
        """Test that to_dict includes source_agent when set."""
        event = AgentEvent.sub_agent_tool_start(
            tool_name="jira_search",
            source_agent="jira-agent",
            input_data={"query": "test"},
        )

        d = event.to_dict()

        assert "source_agent" in d
        assert d["source_agent"] == "jira-agent"

    def test_from_dict_basic(self):
        """Test creating event from dictionary."""
        d = {
            "type": "tool_call_start",
            "name": "search",
            "run_id": "run-123",
            "data": {"input": {"query": "test"}},
        }

        event = AgentEvent.from_dict(d)

        assert event.type == "tool_call_start"
        assert event.name == "search"
        assert event.run_id == "run-123"
        assert event.data["input"]["query"] == "test"
        assert event.source_agent == ""

    def test_from_dict_with_source_agent(self):
        """Test creating event from dictionary with source_agent."""
        d = {
            "type": "sub_agent_tool_end",
            "name": "search",
            "source_agent": "faq-agent",
            "data": {"output": "result"},
        }

        event = AgentEvent.from_dict(d)

        assert event.type == "sub_agent_tool_end"
        assert event.source_agent == "faq-agent"

    def test_roundtrip_serialization(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = AgentEvent.sub_agent_tool_start(
            tool_name="analyze",
            source_agent="analyzer",
            input_data={"doc": "test.pdf"},
            run_id="run-456",
        )

        d = original.to_dict()
        restored = AgentEvent.from_dict(d)

        assert restored.type == original.type
        assert restored.name == original.name
        assert restored.run_id == original.run_id
        assert restored.source_agent == original.source_agent
        assert restored.data == original.data
