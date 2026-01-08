"""MASK A2A Pipe for Open WebUI - Streaming Version.

This Pipe Function connects Open WebUI directly to MASK A2A servers
using SSE streaming for real-time responses.

Features:
- Direct A2A JSON-RPC communication (no OpenAI compat layer)
- SSE streaming support with iter_lines parsing
- chat_id -> contextId mapping for session tracking

Installation:
1. Open WebUI Admin -> Functions -> Add Function
2. Paste this code
3. Enable the function

Configuration:
- A2A_ENDPOINT: Your A2A server URL (use host.docker.internal for Docker)
- TIMEOUT: Request timeout in seconds
- DEBUG: Enable debug logging
"""

import json
import uuid
from typing import Any, Dict, Generator, Optional

import requests
from pydantic import BaseModel, Field


class Pipe:
    """MASK A2A Pipe with streaming support."""

    class Valves(BaseModel):
        """Configuration options."""

        A2A_ENDPOINT: str = Field(
            default="http://host.docker.internal:10002",
            description="A2A server endpoint (use host.docker.internal for Docker)",
        )
        TIMEOUT: int = Field(
            default=120,
            description="Request timeout in seconds",
        )
        DEBUG: bool = Field(
            default=True,
            description="Enable debug logging",
        )

    def __init__(self):
        """Initialize the pipe."""
        self.valves = self.Valves()

    def pipes(self):
        """Return available models."""
        return [{"id": "mask-a2a", "name": "MASK A2A Agent"}]

    def pipe(
        self,
        body: Dict[str, Any],
        __user__: Optional[Dict[str, Any]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Handle chat requests with streaming.

        Args:
            body: Request body with messages.
            __user__: User info from Open WebUI.
            __metadata__: Request metadata including chat_id.

        Yields:
            Response text chunks.
        """
        messages = body.get("messages", [])
        if not messages:
            yield "No messages."
            return

        # Extract chat_id for session tracking
        metadata = __metadata__ or {}
        chat_id = metadata.get("chat_id") or body.get("chat_id")
        user_message = messages[-1].get("content", "")

        if self.valves.DEBUG:
            print(f"[MASK A2A] Streaming request to {self.valves.A2A_ENDPOINT}")
            print(f"[MASK A2A] chat_id={chat_id}")

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

            # Parse SSE stream using iter_lines
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                        text = self._extract_text(data)
                        if text:
                            yield text
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            yield f"Error: {e}"

    def _extract_text(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract text from SSE event data.

        A2A streaming events have structure:
        {
            "result": {
                "artifact": {
                    "parts": [{"kind": "text", "text": "..."}]
                },
                "kind": "artifact-update"
            }
        }

        Args:
            data: Parsed SSE event data.

        Returns:
            Extracted text or None.
        """
        result = data.get("result", {})

        # Handle artifact-update events
        if "artifact" in result:
            parts = result["artifact"].get("parts", [])
            for part in parts:
                if isinstance(part, dict) and part.get("kind") == "text":
                    return part.get("text", "")

        return None
