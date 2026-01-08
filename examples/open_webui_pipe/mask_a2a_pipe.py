"""MASK A2A Pipe Function for Open WebUI.

This Pipe Function connects Open WebUI directly to MASK A2A servers,
implementing the "Frontend Source of Truth" pattern for session sync.

Features:
- Direct A2A JSON-RPC communication (no OpenAI compat layer)
- SSE streaming support for real-time responses
- Full message history passthrough for sync detection
- Metadata injection for checkpoint tracking

Installation:
1. Copy this file to Open WebUI's Functions section
2. Configure the A2A endpoint via Valves
3. Enable the pipe for your workspace

Usage:
- The pipe sends complete message history to enable backend sync
- Backend uses diff detection for regenerate/deletion handling
- Response metadata contains checkpoint_id for retry support
"""

import json
import logging
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Pipe:
    """MASK A2A Pipe for Open WebUI.

    Connects directly to MASK A2A servers using JSON-RPC protocol.
    Implements Frontend Source of Truth pattern for session synchronization.
    """

    class Valves(BaseModel):
        """Configuration options for the pipe."""

        A2A_ENDPOINT: str = Field(
            default="http://localhost:10001",
            description="MASK A2A server endpoint URL",
        )
        TIMEOUT: int = Field(
            default=120,
            description="Request timeout in seconds",
        )
        SHOW_THINKING: bool = Field(
            default=False,
            description="Show agent thinking/tool calls (if supported)",
        )
        ENABLE_STREAMING: bool = Field(
            default=True,
            description="Enable SSE streaming responses",
        )
        DEBUG: bool = Field(
            default=False,
            description="Enable debug logging",
        )

    def __init__(self):
        """Initialize the pipe."""
        self.valves = self.Valves()
        self._session: Optional[requests.Session] = None

    @property
    def session(self) -> requests.Session:
        """Get or create HTTP session."""
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def pipes(self) -> List[Dict[str, str]]:
        """Return available pipe models.

        Open WebUI calls this to get the list of available "models"
        that this pipe provides.
        """
        return [
            {
                "id": "mask-a2a",
                "name": "MASK A2A Agent",
            }
        ]

    def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Generator[str, None, None]]:
        """Main pipe function - handle chat requests.

        Args:
            body: Request body containing messages and options.
            __user__: User information from Open WebUI.
            __metadata__: Request metadata including chat_id.

        Returns:
            Response string or generator for streaming.
        """
        if self.valves.DEBUG:
            logger.info("Pipe called with body: %s", json.dumps(body, indent=2))
            logger.info("Metadata: %s", json.dumps(__metadata__ or {}, indent=2))

        # Extract messages from request body
        messages = body.get("messages", [])
        if not messages:
            return "No messages provided."

        # Extract chat_id as context_id for A2A
        # This is the key ID mapping: chat_id = context_id = thread_id
        metadata = __metadata__ or {}
        chat_id = metadata.get("chat_id") or body.get("chat_id")
        user_id = (__user__ or {}).get("id")

        # Get the latest user message
        user_message = messages[-1].get("content", "") if messages else ""

        # Build A2A request
        # Key: Send full message history for backend sync detection
        a2a_request = self._build_a2a_request(
            user_message=user_message,
            context_id=chat_id,
            full_history=messages,
            user_id=user_id,
        )

        # Execute request
        if self.valves.ENABLE_STREAMING:
            return self._stream_request(a2a_request)
        else:
            return self._sync_request(a2a_request)

    def _build_a2a_request(
        self,
        user_message: str,
        context_id: Optional[str],
        full_history: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build A2A JSON-RPC request.

        Args:
            user_message: The current user message.
            context_id: Chat/context ID for session tracking.
            full_history: Complete message history for sync detection.
            user_id: Optional user ID.

        Returns:
            A2A JSON-RPC request dictionary.
        """
        # Build message with metadata
        message = {
            "role": "user",
            "parts": [{"kind": "text", "text": user_message}],
        }

        if context_id:
            message["contextId"] = context_id

        # Build metadata with full history for sync
        # This enables backend to detect regenerate/deletion
        message["metadata"] = {
            "configuration": {
                "fullHistory": full_history,  # Complete history for diff
            },
        }

        if user_id:
            message["metadata"]["userId"] = user_id

        # A2A JSON-RPC request
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/stream" if self.valves.ENABLE_STREAMING else "message/send",
            "params": {
                "message": message,
            },
        }

    def _sync_request(self, request: Dict[str, Any]) -> str:
        """Send synchronous (non-streaming) request.

        Args:
            request: A2A JSON-RPC request.

        Returns:
            Response text.
        """
        try:
            response = self.session.post(
                self.valves.A2A_ENDPOINT,
                json=request,
                timeout=self.valves.TIMEOUT,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            result = response.json()
            if self.valves.DEBUG:
                logger.info("Response: %s", json.dumps(result, indent=2))

            # Extract response from A2A result
            return self._extract_response_text(result)

        except requests.exceptions.RequestException as e:
            logger.exception("A2A request failed: %s", e)
            return f"Error connecting to A2A server: {e}"
        except Exception as e:
            logger.exception("Unexpected error: %s", e)
            return f"Error: {e}"

    def _stream_request(
        self, request: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Send streaming request and yield chunks.

        Args:
            request: A2A JSON-RPC request.

        Yields:
            Response text chunks.
        """
        try:
            response = self.session.post(
                self.valves.A2A_ENDPOINT,
                json=request,
                timeout=self.valves.TIMEOUT,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                stream=True,
            )
            response.raise_for_status()

            # Parse SSE events using iter_lines for reliable parsing
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    data_str = line[5:].strip()
                    event_data = self._parse_sse_event(data_str)
                    if event_data:
                        text = self._extract_text_from_event(event_data)
                        if text:
                            yield text

        except requests.exceptions.RequestException as e:
            logger.exception("A2A streaming request failed: %s", e)
            yield f"Error connecting to A2A server: {e}"
        except Exception as e:
            logger.exception("Unexpected streaming error: %s", e)
            yield f"Error: {e}"

    def _parse_sse_stream(
        self, response: requests.Response
    ) -> Iterator[str]:
        """Parse Server-Sent Events stream (legacy buffer-based method).

        Note: Prefer using iter_lines() in _stream_request for reliability.

        Args:
            response: HTTP response with SSE stream.

        Yields:
            Text chunks from SSE events.
        """
        buffer = ""

        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if not chunk:
                continue

            buffer += chunk

            # Process complete SSE events
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                event_data = self._parse_sse_event(event_str)

                if event_data:
                    text = self._extract_text_from_event(event_data)
                    if text:
                        yield text

    def _parse_sse_event(self, event_str: str) -> Optional[Dict[str, Any]]:
        """Parse a single SSE event.

        Handles both raw JSON strings and full SSE event strings.

        Args:
            event_str: Raw JSON string or full SSE event string.

        Returns:
            Parsed event data or None.
        """
        # Try parsing as direct JSON first (for iter_lines usage)
        try:
            data = json.loads(event_str)
            return {"type": None, "data": data}
        except json.JSONDecodeError:
            pass

        # Fall back to parsing full SSE event format
        event_type = None
        data_lines = []

        for line in event_str.split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if not data_lines:
            return None

        try:
            data = json.loads("".join(data_lines))
            return {"type": event_type, "data": data}
        except json.JSONDecodeError:
            if self.valves.DEBUG:
                logger.warning("Failed to parse SSE data: %s", data_lines)
            return None

    def _extract_text_from_event(self, event: Dict[str, Any]) -> Optional[str]:
        """Extract text content from SSE event.

        Handles various A2A event types:
        - TaskArtifactUpdateEvent: Streaming text chunks
        - AgentTextMessage: Complete messages

        Args:
            event: Parsed SSE event.

        Returns:
            Extracted text or None.
        """
        data = event.get("data", {})
        event_type = event.get("type", "")

        # Handle different A2A event types
        if event_type == "TaskArtifactUpdateEvent" or "artifact" in data:
            # Streaming chunk
            artifact = data.get("artifact", {})
            parts = artifact.get("parts", [])
            for part in parts:
                if isinstance(part, dict):
                    # Part can be nested in "root"
                    root = part.get("root", part)
                    if "text" in root:
                        return root["text"]
                elif isinstance(part, str):
                    return part

        elif "message" in data:
            # Complete message response
            message = data.get("message", {})
            parts = message.get("parts", [])
            for part in parts:
                if isinstance(part, dict):
                    root = part.get("root", part)
                    if "text" in root:
                        return root["text"]

        elif "result" in data:
            # JSON-RPC result
            result = data.get("result", {})
            if isinstance(result, str):
                return result
            if "message" in result:
                return self._extract_message_text(result["message"])

        return None

    def _extract_response_text(self, result: Dict[str, Any]) -> str:
        """Extract response text from JSON-RPC result.

        Args:
            result: JSON-RPC response.

        Returns:
            Extracted text.
        """
        if "error" in result:
            error = result["error"]
            return f"Error: {error.get('message', 'Unknown error')}"

        data = result.get("result", {})

        # Handle message response
        if "message" in data:
            return self._extract_message_text(data["message"])

        # Handle artifacts
        if "artifacts" in data:
            texts = []
            for artifact in data["artifacts"]:
                for part in artifact.get("parts", []):
                    root = part.get("root", part)
                    if "text" in root:
                        texts.append(root["text"])
            return "".join(texts)

        return str(data)

    def _extract_message_text(self, message: Dict[str, Any]) -> str:
        """Extract text from A2A message.

        Args:
            message: A2A message dictionary.

        Returns:
            Extracted text.
        """
        parts = message.get("parts", [])
        texts = []

        for part in parts:
            if isinstance(part, dict):
                root = part.get("root", part)
                if "text" in root:
                    texts.append(root["text"])
            elif isinstance(part, str):
                texts.append(part)

        return "".join(texts)


# For Open WebUI function registry
def get_pipe():
    """Factory function for Open WebUI."""
    return Pipe()
