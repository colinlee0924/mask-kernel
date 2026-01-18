"""Agent event types for structured streaming.

This module defines event types emitted during agent execution,
enabling real-time streaming of tool calls, LLM responses, and
agent lifecycle events.

These events are consumed by adapters (e.g., OpenAI adapter) to
provide structured streaming output for UI rendering.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

# Event type literals for type safety
EventType = Literal[
    # Agent lifecycle events
    "agent_start",      # Agent execution started
    "agent_end",        # Agent execution completed
    # LLM events
    "text_delta",       # LLM text token (incremental)
    "thinking_start",   # Reasoning/thinking phase started
    "thinking_end",     # Reasoning/thinking phase ended
    # Tool events
    "tool_call_start",  # Tool execution started
    "tool_call_end",    # Tool execution completed
    # Delegation events (orchestrator -> sub-agent)
    "delegation_start",       # Orchestrator delegating to sub-agent
    "delegation_end",         # Sub-agent completed delegation
    # Sub-agent events (forwarded from sub-agent)
    "sub_agent_tool_start",   # Sub-agent tool execution started
    "sub_agent_tool_end",     # Sub-agent tool execution completed
    "sub_agent_text_delta",   # Sub-agent text token
    "sub_agent_thinking",     # Sub-agent thinking phase
    # Error
    "error",            # Error occurred
    "sub_agent_error",  # Sub-agent error
]


@dataclass
class AgentEvent:
    """Structured event emitted during agent execution.

    Attributes:
        type: Event type identifier.
        data: Event-specific payload data.
        name: Name of the runnable/tool that generated this event.
        run_id: Unique identifier for this execution run.
        source_agent: Name of the agent that generated this event (for sub-agent events).

    Example:
        # Tool call start event
        event = AgentEvent(
            type="tool_call_start",
            name="search_jira",
            run_id="abc123",
            data={"input": {"query": "open bugs"}}
        )

        # Text delta event
        event = AgentEvent(
            type="text_delta",
            data={"delta": "Here are the results..."}
        )

        # Sub-agent tool event
        event = AgentEvent(
            type="sub_agent_tool_start",
            name="jira_search",
            source_agent="jira-agent",
            data={"input": {"query": "AI bugs"}}
        )
    """

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    run_id: str = ""
    source_agent: str = ""  # Name of the source agent for sub-agent events

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        result = {
            "type": self.type,
            "name": self.name,
            "run_id": self.run_id,
            "data": self.data,
        }
        if self.source_agent:
            result["source_agent"] = self.source_agent
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentEvent":
        """Create event from dictionary."""
        return cls(
            type=d["type"],
            name=d.get("name", ""),
            run_id=d.get("run_id", ""),
            data=d.get("data", {}),
            source_agent=d.get("source_agent", ""),
        )

    @classmethod
    def text_delta(cls, content: str, run_id: str = "") -> "AgentEvent":
        """Create a text delta event."""
        return cls(
            type="text_delta",
            data={"delta": content},
            run_id=run_id,
        )

    @classmethod
    def tool_start(
        cls,
        name: str,
        run_id: str,
        input_data: Dict[str, Any],
    ) -> "AgentEvent":
        """Create a tool call start event."""
        return cls(
            type="tool_call_start",
            name=name,
            run_id=run_id,
            data={"input": input_data},
        )

    @classmethod
    def tool_end(
        cls,
        name: str,
        run_id: str,
        output: str,
    ) -> "AgentEvent":
        """Create a tool call end event."""
        return cls(
            type="tool_call_end",
            name=name,
            run_id=run_id,
            data={"output": output},
        )

    @classmethod
    def agent_start(cls, name: str = "", run_id: str = "") -> "AgentEvent":
        """Create an agent start event."""
        return cls(
            type="agent_start",
            name=name,
            run_id=run_id,
        )

    @classmethod
    def agent_end(cls, name: str = "", run_id: str = "") -> "AgentEvent":
        """Create an agent end event."""
        return cls(
            type="agent_end",
            name=name,
            run_id=run_id,
        )

    @classmethod
    def error(cls, message: str, run_id: str = "") -> "AgentEvent":
        """Create an error event."""
        return cls(
            type="error",
            run_id=run_id,
            data={"message": message},
        )

    # =========================================================================
    # Delegation Events (orchestrator -> sub-agent)
    # =========================================================================

    @classmethod
    def delegation_start(
        cls,
        target_agent: str,
        task: str,
        run_id: str = "",
    ) -> "AgentEvent":
        """Create a delegation start event.

        Args:
            target_agent: Name of the sub-agent being delegated to.
            task: Task description being delegated.
            run_id: Unique identifier for this delegation.
        """
        return cls(
            type="delegation_start",
            name=target_agent,
            run_id=run_id,
            data={"task": task, "target_agent": target_agent},
        )

    @classmethod
    def delegation_end(
        cls,
        target_agent: str,
        result: str,
        run_id: str = "",
        success: bool = True,
    ) -> "AgentEvent":
        """Create a delegation end event.

        Args:
            target_agent: Name of the sub-agent that completed.
            result: Result from the sub-agent.
            run_id: Unique identifier for this delegation.
            success: Whether the delegation completed successfully.
        """
        return cls(
            type="delegation_end",
            name=target_agent,
            run_id=run_id,
            data={"result": result, "target_agent": target_agent, "success": success},
        )

    # =========================================================================
    # Sub-agent Events (forwarded from sub-agent with source_agent set)
    # =========================================================================

    @classmethod
    def from_sub_agent(
        cls,
        event: "AgentEvent",
        source_agent: str,
    ) -> "AgentEvent":
        """Convert a regular event to a sub-agent event.

        This transforms events received from a sub-agent's stream into
        properly typed sub-agent events with source_agent set.

        Args:
            event: Original event from sub-agent.
            source_agent: Name of the sub-agent that generated the event.

        Returns:
            New AgentEvent with sub_agent_ prefix and source_agent set.
        """
        # Map regular event types to sub-agent event types
        type_mapping = {
            "tool_call_start": "sub_agent_tool_start",
            "tool_call_end": "sub_agent_tool_end",
            "text_delta": "sub_agent_text_delta",
            "thinking_start": "sub_agent_thinking",
            "thinking_end": "sub_agent_thinking",
            "error": "sub_agent_error",
        }

        new_type = type_mapping.get(event.type, event.type)

        return cls(
            type=new_type,  # type: ignore
            name=event.name,
            run_id=event.run_id,
            data=event.data,
            source_agent=source_agent,
        )

    @classmethod
    def sub_agent_tool_start(
        cls,
        tool_name: str,
        source_agent: str,
        input_data: Dict[str, Any],
        run_id: str = "",
    ) -> "AgentEvent":
        """Create a sub-agent tool start event."""
        return cls(
            type="sub_agent_tool_start",
            name=tool_name,
            run_id=run_id,
            source_agent=source_agent,
            data={"input": input_data},
        )

    @classmethod
    def sub_agent_tool_end(
        cls,
        tool_name: str,
        source_agent: str,
        output: str,
        run_id: str = "",
        duration_ms: int = 0,
    ) -> "AgentEvent":
        """Create a sub-agent tool end event."""
        return cls(
            type="sub_agent_tool_end",
            name=tool_name,
            run_id=run_id,
            source_agent=source_agent,
            data={"output": output, "duration_ms": duration_ms},
        )

    @classmethod
    def sub_agent_text_delta(
        cls,
        content: str,
        source_agent: str,
        run_id: str = "",
    ) -> "AgentEvent":
        """Create a sub-agent text delta event."""
        return cls(
            type="sub_agent_text_delta",
            source_agent=source_agent,
            run_id=run_id,
            data={"delta": content},
        )

    @classmethod
    def sub_agent_error(
        cls,
        message: str,
        source_agent: str,
        run_id: str = "",
    ) -> "AgentEvent":
        """Create a sub-agent error event."""
        return cls(
            type="sub_agent_error",
            source_agent=source_agent,
            run_id=run_id,
            data={"message": message},
        )
