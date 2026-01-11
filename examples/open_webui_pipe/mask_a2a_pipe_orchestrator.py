"""MASK A2A Pipe for Open WebUI - Orchestrator Version with Sub-Agent Events.

This Pipe Function connects Open WebUI to a MASK orchestrator agent that
delegates tasks to sub-agents, showing real-time events from all agents.

Features:
- Real-time streaming of orchestrator thinking process
- Sub-agent delegation visualization (nested events)
- Tool call visualization for both orchestrator and sub-agents
- Collapsible sections for complex multi-agent responses

Event Types:
- thinking: Orchestrator reasoning process
- tool_call: Orchestrator tool invocation
- tool_result: Tool execution result
- response: Final text response
- sub_agent_delegation: Delegation to sub-agent
- sub_agent_tool_call: Sub-agent tool invocation
- sub_agent_tool_result: Sub-agent tool result
- sub_agent_response: Sub-agent text output
- sub_agent_thinking: Sub-agent reasoning

Installation:
1. Open WebUI Admin -> Functions -> Add Function
2. Paste this code
3. Enable the function

Configuration:
- A2A_ENDPOINT: Your orchestrator A2A server URL
- SHOW_THINKING: Whether to show thinking process (default True)
- SHOW_TOOL_CALLS: Whether to show tool calls (default True)
- SHOW_SUB_AGENTS: Whether to show sub-agent events (default True)
"""

import json
import uuid
from typing import Any, Dict, Generator, Optional

import requests
from pydantic import BaseModel, Field


class Pipe:
    """MASK A2A Orchestrator Pipe with sub-agent event support."""

    class Valves(BaseModel):
        """Configuration options."""

        A2A_ENDPOINT: str = Field(
            default="http://host.docker.internal:10001",
            description="Orchestrator A2A server endpoint",
        )
        TIMEOUT: int = Field(
            default=180,
            description="Request timeout in seconds (longer for multi-agent)",
        )
        SHOW_THINKING: bool = Field(
            default=True,
            description="Show agent thinking process",
        )
        SHOW_TOOL_CALLS: bool = Field(
            default=True,
            description="Show tool calls and results",
        )
        SHOW_SUB_AGENTS: bool = Field(
            default=True,
            description="Show sub-agent events",
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
        self._in_sub_agent = False
        self._thinking_started = False
        self._current_tool = None
        self._current_sub_agent = None

    def pipes(self):
        """Return available models."""
        return [{"id": "mask-orchestrator", "name": "MASK Orchestrator Agent"}]

    def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Handle chat requests with multi-agent streaming.

        Args:
            body: Request body with messages.
            __user__: User info from Open WebUI.
            __metadata__: Request metadata including chat_id.

        Yields:
            Formatted response chunks with HTML for multi-agent rendering.
        """
        messages = body.get("messages", [])
        if not messages:
            yield "No messages."
            return

        # Reset state
        self._reset_state()

        # Extract chat_id for session tracking
        metadata = __metadata__ or {}
        chat_id = metadata.get("chat_id") or body.get("chat_id")
        user_message = messages[-1].get("content", "")

        if self.valves.DEBUG:
            print(f"[MASK Orchestrator] Request to {self.valves.A2A_ENDPOINT}")
            print(f"[MASK Orchestrator] chat_id={chat_id}")

        # Build A2A message
        message = {
            "role": "user",
            "parts": [{"kind": "text", "text": user_message}],
            "messageId": str(uuid.uuid4()),
        }
        if chat_id:
            message["contextId"] = chat_id

        # Build JSON-RPC request for sendSubscribe
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/sendSubscribe",
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

    def _reset_state(self):
        """Reset rendering state."""
        self._in_thinking = False
        self._in_tool_call = False
        self._in_sub_agent = False
        self._thinking_started = False
        self._current_tool = None
        self._current_sub_agent = None

    def _process_event(self, data: Dict[str, Any]) -> Generator[str, None, None]:
        """Process a single SSE event and yield formatted chunks.

        Args:
            data: Parsed SSE event data.

        Yields:
            Formatted text chunks.
        """
        result = data.get("result", {})

        # Handle TaskStatusUpdateEvent (contains status messages)
        if "state" in result:
            for chunk in self._process_status_event(result):
                yield chunk
            return

        # Handle TaskArtifactUpdateEvent (contains artifacts)
        if "artifact" not in result:
            return

        artifact = result["artifact"]
        artifact_name = artifact.get("name", "response")
        parts = artifact.get("parts", [])

        for part in parts:
            if not isinstance(part, dict):
                continue

            text = self._extract_text(part)
            if not text:
                continue

            # Route to appropriate renderer based on artifact name
            for chunk in self._render_artifact(artifact_name, text, part):
                yield chunk

    def _process_status_event(self, result: Dict[str, Any]) -> Generator[str, None, None]:
        """Process TaskStatusUpdateEvent for status updates.

        Args:
            result: Status event result.

        Yields:
            Formatted status chunks.
        """
        message = result.get("message", {})
        parts = message.get("parts", [])

        for part in parts:
            if not isinstance(part, dict):
                continue

            # Check for structured event data
            if "data" in part:
                event_data = part["data"]
                event_type = event_data.get("event_type", "")

                for chunk in self._render_status_event(event_type, event_data):
                    yield chunk

    def _render_status_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
    ) -> Generator[str, None, None]:
        """Render status event based on type.

        Args:
            event_type: Type of the event.
            event_data: Event data dict.

        Yields:
            Formatted chunks.
        """
        source_agent = event_data.get("source_agent", "")
        agent_name = event_data.get("agent_name", "")
        tool_name = event_data.get("tool_name", "")

        # Sub-agent events
        if event_type.startswith("sub_agent_"):
            if not self.valves.SHOW_SUB_AGENTS:
                return

            if event_type == "sub_agent_tool_start":
                yield f"\n> 🔧 **[{source_agent}]** `{tool_name}` running...\n"
            elif event_type == "sub_agent_tool_end":
                duration = event_data.get("duration_ms", 0)
                yield f"> ✅ **[{source_agent}]** `{tool_name}` done ({duration}ms)\n"

        # Orchestrator events
        elif event_type == "tool_start":
            if self.valves.SHOW_TOOL_CALLS:
                yield f"\n🔧 **{tool_name}** running...\n"
        elif event_type == "tool_end":
            if self.valves.SHOW_TOOL_CALLS:
                duration = event_data.get("duration_ms", 0)
                yield f"✅ **{tool_name}** done ({duration}ms)\n"
        elif event_type == "tool_decision":
            tools = event_data.get("tools", [])
            if tools:
                yield f"\n💡 Decided to call: {', '.join(tools)}\n"
        elif event_type == "llm_thinking":
            if self.valves.SHOW_THINKING:
                yield f"\n🤔 Thinking...\n"

    def _extract_text(self, part: Dict[str, Any]) -> Optional[str]:
        """Extract text from a part object.

        Args:
            part: Part dict from artifact.

        Returns:
            Extracted text or None.
        """
        if part.get("kind") == "text":
            return part.get("text", "")
        elif "root" in part:
            root = part["root"]
            if isinstance(root, dict):
                return root.get("text", "")
        elif "text" in part:
            return part["text"]
        return None

    def _render_artifact(
        self,
        artifact_name: str,
        text: str,
        part: Dict[str, Any],
    ) -> Generator[str, None, None]:
        """Render artifact based on name.

        Args:
            artifact_name: Name of the artifact.
            text: Text content.
            part: Full part dict.

        Yields:
            Formatted chunks.
        """
        # Sub-agent artifacts
        if artifact_name.startswith("sub_agent_"):
            for chunk in self._render_sub_agent_artifact(artifact_name, text, part):
                yield chunk
            return

        # Orchestrator artifacts
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

    def _render_sub_agent_artifact(
        self,
        artifact_name: str,
        text: str,
        part: Dict[str, Any],
    ) -> Generator[str, None, None]:
        """Render sub-agent specific artifacts.

        Args:
            artifact_name: Sub-agent artifact name.
            text: Text content.
            part: Full part dict.

        Yields:
            Formatted chunks with indentation.
        """
        if not self.valves.SHOW_SUB_AGENTS:
            return

        if artifact_name == "sub_agent_delegation":
            try:
                data = json.loads(text)
                target = data.get("target_agent", "sub-agent")
                task = data.get("task", "")
                if data.get("success") is False:
                    yield f"\n❌ **{target}** failed\n"
                elif "result" in data:
                    yield f"\n✅ **{target}** completed\n"
                else:
                    yield f"\n📤 Delegating to **{target}**...\n"
            except json.JSONDecodeError:
                yield f"\n📤 Delegation: {text}\n"

        elif artifact_name == "sub_agent_tool_call":
            try:
                data = json.loads(text)
                tool = data.get("tool", "tool")
                input_data = data.get("input", {})
                yield f"\n> 🔧 **{tool}**\n"
                yield f"> ```json\n> {json.dumps(input_data, indent=2)}\n> ```\n"
            except json.JSONDecodeError:
                yield f"\n> 🔧 {text}\n"

        elif artifact_name == "sub_agent_tool_result":
            try:
                data = json.loads(text)
                tool = data.get("tool", "tool")
                output = data.get("output", text)
                duration = data.get("duration_ms", 0)
                yield f"\n> ✅ **{tool}** ({duration}ms)\n"
                if len(output) > 200:
                    yield f"> <details><summary>Output</summary>\n>\n> {output}\n>\n> </details>\n"
                else:
                    yield f"> {output}\n"
            except json.JSONDecodeError:
                yield f"\n> ✅ {text}\n"

        elif artifact_name == "sub_agent_response":
            yield f"> {text}"

        elif artifact_name == "sub_agent_thinking":
            if self.valves.SHOW_THINKING:
                yield f"> 🤔 {text}\n"

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
            yield "\n<details open>\n<summary>🧠 Thinking</summary>\n\n"

        yield text

    def _render_tool_call(self, text: str) -> Generator[str, None, None]:
        """Render tool call with syntax highlighting.

        Args:
            text: Tool call JSON text.

        Yields:
            Formatted tool call chunks.
        """
        if not self.valves.SHOW_TOOL_CALLS:
            return

        # Close thinking if open
        if self._in_thinking:
            self._in_thinking = False
            yield "\n\n</details>\n\n"

        try:
            data = json.loads(text)
            tool_name = data.get("tool", "unknown")
            tool_input = data.get("input", {})

            self._current_tool = tool_name
            self._in_tool_call = True

            yield f"\n<details open>\n<summary>🔧 {tool_name}</summary>\n\n"
            yield f"**Input:**\n```json\n{json.dumps(tool_input, indent=2)}\n```\n"

        except json.JSONDecodeError:
            yield f"\n🔧 Tool call: {text}\n"

    def _render_tool_result(self, text: str) -> Generator[str, None, None]:
        """Render tool result.

        Args:
            text: Tool result text.

        Yields:
            Formatted tool result chunks.
        """
        if not self.valves.SHOW_TOOL_CALLS:
            return

        if self._in_tool_call:
            # Large results get collapsible
            if len(text) > 500:
                yield f"\n**Output:**\n<details><summary>Show result</summary>\n\n```\n{text}\n```\n\n</details>\n"
            else:
                yield f"\n**Output:**\n```\n{text}\n```\n"

            yield "\n</details>\n\n"
            self._in_tool_call = False
            self._current_tool = None
        else:
            yield f"\n**Result:** {text}\n"

    def _close_sections(self) -> Generator[str, None, None]:
        """Close any open HTML sections.

        Yields:
            Closing tags for open sections.
        """
        if self._in_tool_call:
            yield "\n</details>\n"
            self._in_tool_call = False

        if self._in_thinking:
            yield "\n</details>\n"
            self._in_thinking = False

        if self._in_sub_agent:
            yield "\n</details>\n"
            self._in_sub_agent = False
