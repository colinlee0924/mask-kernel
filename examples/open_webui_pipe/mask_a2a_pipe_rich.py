"""MASK A2A Pipe for Open WebUI - Rich Streaming Version.

This Pipe Function connects Open WebUI to MASK A2A servers with
rich event streaming support for thinking, tool calls, and results.

Features:
- Real-time streaming of agent thinking process
- Tool call visualization with input/output
- Collapsible sections for complex responses
- Direct A2A JSON-RPC communication

Event Types:
- thinking: Model reasoning process (rendered in collapsible)
- tool_call: Tool invocation with input (rendered with syntax highlighting)
- tool_result: Tool execution result
- response: Final text response (streamed as-is)

Installation:
1. Open WebUI Admin -> Functions -> Add Function
2. Paste this code
3. Enable the function

Configuration:
- A2A_ENDPOINT: Your A2A server URL
- SHOW_THINKING: Whether to show thinking process (default True)
- SHOW_TOOL_CALLS: Whether to show tool calls (default True)
"""

import json
import uuid
from typing import Any, Dict, Generator, Optional

import requests
from pydantic import BaseModel, Field


class Pipe:
    """MASK A2A Pipe with rich streaming support."""

    class Valves(BaseModel):
        """Configuration options."""

        A2A_ENDPOINT: str = Field(
            default="http://host.docker.internal:10002",
            description="A2A server endpoint",
        )
        TIMEOUT: int = Field(
            default=120,
            description="Request timeout in seconds",
        )
        SHOW_THINKING: bool = Field(
            default=True,
            description="Show agent thinking process",
        )
        SHOW_TOOL_CALLS: bool = Field(
            default=True,
            description="Show tool calls and results",
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging",
        )

    def __init__(self):
        """Initialize the pipe."""
        self.valves = self.Valves()
        # Track state for rendering
        self._in_thinking = False
        self._in_tool_call = False
        self._thinking_started = False
        self._current_tool = None

    def pipes(self):
        """Return available models."""
        return [{"id": "mask-a2a-rich", "name": "MASK A2A Agent (Rich)"}]

    def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Handle chat requests with rich streaming.

        Args:
            body: Request body with messages.
            __user__: User info from Open WebUI.
            __metadata__: Request metadata including chat_id.

        Yields:
            Formatted response chunks with HTML tags for rendering.
        """
        messages = body.get("messages", [])
        if not messages:
            yield "No messages."
            return

        # Reset state
        self._in_thinking = False
        self._in_tool_call = False
        self._thinking_started = False
        self._current_tool = None

        # Extract chat_id for session tracking
        metadata = __metadata__ or {}
        chat_id = metadata.get("chat_id") or body.get("chat_id")
        user_message = messages[-1].get("content", "")

        if self.valves.DEBUG:
            print(f"[MASK A2A Rich] Request to {self.valves.A2A_ENDPOINT}")
            print(f"[MASK A2A Rich] chat_id={chat_id}")

        # Build A2A message
        message = {
            "role": "user",
            "parts": [{"kind": "text", "text": user_message}],
            "messageId": str(uuid.uuid4()),
        }
        if chat_id:
            message["contextId"] = chat_id

        # Build JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/stream",
            "params": {"message": message},
        }

        try:
            response = requests.post(
                self.valves.A2A_ENDPOINT,
                json=request,
                timeout=self.valves.TIMEOUT,
                headers={"Accept": "text/event-stream"},
                stream=True,
            )
            response.raise_for_status()

            # Parse SSE stream
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                        for chunk in self._process_event(data):
                            if chunk:
                                yield chunk
                    except json.JSONDecodeError:
                        pass

            # Close any open sections
            for chunk in self._close_sections():
                yield chunk

        except Exception as e:
            yield f"\n\n**Error:** {e}"

    def _process_event(self, data: Dict[str, Any]) -> Generator[str, None, None]:
        """Process a single SSE event and yield formatted chunks.

        Args:
            data: Parsed SSE event data.

        Yields:
            Formatted text chunks.
        """
        result = data.get("result", {})
        if "artifact" not in result:
            return

        artifact = result["artifact"]
        artifact_name = artifact.get("name", "response")
        parts = artifact.get("parts", [])

        for part in parts:
            if not isinstance(part, dict):
                continue

            text = None
            if part.get("kind") == "text":
                text = part.get("text", "")
            elif "root" in part:
                root = part["root"]
                if isinstance(root, dict):
                    text = root.get("text", "")

            if not text:
                continue

            # Route to appropriate renderer
            if artifact_name == "thinking":
                for chunk in self._render_thinking(text):
                    yield chunk
            elif artifact_name == "tool_call":
                for chunk in self._render_tool_call(text):
                    yield chunk
            elif artifact_name == "tool_result":
                for chunk in self._render_tool_result(text):
                    yield chunk
            else:
                # Normal response - stream as-is
                yield text

    def _render_thinking(self, text: str) -> Generator[str, None, None]:
        """Render thinking content with collapsible section.

        Args:
            text: Thinking text chunk.

        Yields:
            Formatted thinking chunks.
        """
        if not self.valves.SHOW_THINKING:
            return

        if not self._thinking_started:
            self._thinking_started = True
            self._in_thinking = True
            # Open collapsible thinking section
            yield "\n<details>\n<summary>💭 Thinking...</summary>\n\n"

        yield text

    def _render_tool_call(self, text: str) -> Generator[str, None, None]:
        """Render tool call with formatted display.

        Args:
            text: Tool call JSON string.

        Yields:
            Formatted tool call chunks.
        """
        if not self.valves.SHOW_TOOL_CALLS:
            return

        # Close thinking section if open
        if self._in_thinking:
            yield "\n</details>\n\n"
            self._in_thinking = False

        try:
            tool_data = json.loads(text)
            tool_name = tool_data.get("tool", "unknown")
            tool_input = tool_data.get("input", {})
            status = tool_data.get("status", "running")

            self._current_tool = tool_name
            self._in_tool_call = True

            # Format tool call with Open WebUI compatible HTML
            yield f"\n<details open>\n<summary>🔧 {tool_name}"
            if status == "running":
                yield " (running...)"
            yield "</summary>\n\n"
            yield "**Input:**\n```json\n"
            yield json.dumps(tool_input, ensure_ascii=False, indent=2)
            yield "\n```\n"

        except json.JSONDecodeError:
            yield f"\n🔧 Tool call: {text}\n"

    def _render_tool_result(self, text: str) -> Generator[str, None, None]:
        """Render tool result.

        Args:
            text: Tool result JSON string.

        Yields:
            Formatted tool result chunks.
        """
        if not self.valves.SHOW_TOOL_CALLS:
            return

        try:
            result_data = json.loads(text)
            output = result_data.get("output", "")
            status = result_data.get("status", "completed")

            yield "\n**Output:**\n```\n"
            yield output[:2000]  # Truncate long outputs
            if len(output) > 2000:
                yield "\n... (truncated)"
            yield "\n```\n"

            # Close tool call section
            if self._in_tool_call:
                yield "</details>\n\n"
                self._in_tool_call = False
                self._current_tool = None

        except json.JSONDecodeError:
            yield f"\nResult: {text}\n"

    def _close_sections(self) -> Generator[str, None, None]:
        """Close any open HTML sections.

        Yields:
            Closing tags for open sections.
        """
        if self._in_thinking:
            yield "\n</details>\n\n"
            self._in_thinking = False

        if self._in_tool_call:
            yield "</details>\n\n"
            self._in_tool_call = False
