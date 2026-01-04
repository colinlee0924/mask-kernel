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
    "agent_start",      # Agent execution started
    "agent_end",        # Agent execution completed
    "text_delta",       # LLM text token (incremental)
    "tool_call_start",  # Tool execution started
    "tool_call_end",    # Tool execution completed
    "thinking_start",   # Reasoning/thinking phase started
    "thinking_end",     # Reasoning/thinking phase ended
    "error",            # Error occurred
]


@dataclass
class AgentEvent:
    """Structured event emitted during agent execution.

    Attributes:
        type: Event type identifier.
        data: Event-specific payload data.
        name: Name of the runnable/tool that generated this event.
        run_id: Unique identifier for this execution run.

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
    """

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "name": self.name,
            "run_id": self.run_id,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentEvent":
        """Create event from dictionary."""
        return cls(
            type=d["type"],
            name=d.get("name", ""),
            run_id=d.get("run_id", ""),
            data=d.get("data", {}),
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
