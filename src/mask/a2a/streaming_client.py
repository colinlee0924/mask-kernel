"""Streaming A2A client for sub-agent event propagation.

This module provides a streaming client that subscribes to A2A sub-agent
event streams and yields AgentEvent objects for real-time UI updates.

Following the A2A sendSubscribe SSE pattern for streaming responses.
"""

import json
import logging
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver
from a2a.types import AgentCard, Role

from mask.core.events import AgentEvent

logger = logging.getLogger(__name__)


class StreamingA2AClient:
    """Streaming A2A client for subscribing to sub-agent event streams.

    This client uses A2A's sendSubscribe endpoint to receive SSE events
    from sub-agents and converts them to AgentEvent objects.

    Example:
        client = StreamingA2AClient("http://localhost:10001")
        await client.connect()

        async for event in client.send_message_streaming("Query Jira tickets"):
            print(f"{event.type}: {event.data}")

    Attributes:
        base_url: The base URL of the A2A agent.
        agent_name: Name of the remote agent (discovered or provided).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        agent_name: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        """Initialize streaming client.

        Args:
            base_url: Base URL of the A2A agent.
            agent_name: Optional agent name (discovered from AgentCard if not provided).
            timeout: HTTP timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.agent_name = agent_name
        self.timeout = timeout
        self._card: Optional[AgentCard] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> "StreamingA2AClient":
        """Connect to the remote agent and discover its capabilities.

        Returns:
            Self for method chaining.

        Raises:
            httpx.HTTPError: If connection fails.
        """
        self._http_client = httpx.AsyncClient(timeout=self.timeout)

        # Resolve agent card to get metadata
        resolver = A2ACardResolver(self._http_client, self.base_url)
        self._card = await resolver.get_agent_card()

        if not self.agent_name:
            self.agent_name = self._card.name

        logger.info(
            "Connected to streaming agent: %s at %s",
            self.agent_name,
            self.base_url,
        )

        return self

    async def close(self) -> None:
        """Close the HTTP client connection."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> "StreamingA2AClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    @property
    def card(self) -> Optional[AgentCard]:
        """Get the agent's AgentCard metadata."""
        return self._card

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Create a fresh httpx client for the current request.

        This method always creates a new client to avoid event loop mismatch issues
        that occur when the client was created in a different event loop
        (e.g., during asyncio.run() startup) but is now being used
        in a different event loop (e.g., during uvicorn request handling).

        Returns:
            A fresh httpx.AsyncClient instance.
        """
        try:
            # Always create a fresh client for each request to avoid event loop issues
            # The client will be closed after the streaming request completes
            client = httpx.AsyncClient(timeout=self.timeout)
            logger.debug("Created fresh HTTP client for %s", self.base_url)

            # Re-resolve agent card if we don't have it
            if self._card is None:
                resolver = A2ACardResolver(client, self.base_url)
                self._card = await resolver.get_agent_card()
                if not self.agent_name:
                    self.agent_name = self._card.name

            return client

        except Exception as e:
            logger.error("Failed to create client: %s", e)
            raise RuntimeError(f"Failed to connect to {self.base_url}: {e}") from e

    async def send_message_streaming(
        self,
        text: str,
        context_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Send message and stream events from sub-agent.

        Yields AgentEvent objects as they arrive from the sub-agent's
        SSE stream via A2A's sendSubscribe endpoint.

        Args:
            text: Message text to send.
            context_id: Optional context ID for conversation continuity.
            task_id: Optional task ID for task continuation.

        Yields:
            AgentEvent objects representing sub-agent activity.

        Raises:
            RuntimeError: If not connected.
            httpx.HTTPError: If request fails.
        """
        # Create a fresh client for this request to avoid event loop issues
        client = await self._ensure_client()

        # Generate IDs
        message_id = str(uuid4())
        context_id = context_id or str(uuid4())
        run_id = str(uuid4())

        # Build JSON-RPC request for sendSubscribe
        request_body = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "tasks/sendSubscribe",
            "params": {
                "message": {
                    "messageId": message_id,
                    "contextId": context_id,
                    "taskId": task_id,
                    "role": Role.user.value,
                    "parts": [{"text": text}],
                }
            },
        }

        logger.debug(
            "Sending streaming request to %s: %s...",
            self.agent_name,
            text[:50],
        )

        # Emit delegation start event
        yield AgentEvent.delegation_start(
            target_agent=self.agent_name or "unknown",
            task=text,
            run_id=run_id,
        )

        url = f"{self.base_url}"
        success = True
        final_result = ""

        try:
            async with client.stream(
                "POST",
                url,
                json=request_body,
                headers={
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                },
            ) as response:
                response.raise_for_status()

                # Parse SSE stream
                async for event in self._parse_sse_stream(response):
                    # Convert to AgentEvent with source_agent
                    agent_event = self._convert_to_agent_event(event, run_id)
                    if agent_event:
                        yield agent_event

                        # Track final result from text deltas
                        if agent_event.type == "sub_agent_text_delta":
                            delta = agent_event.data.get("delta", "")
                            final_result += delta

        except Exception as e:
            logger.error("Streaming error from %s: %s", self.agent_name, e)
            success = False
            yield AgentEvent.sub_agent_error(
                message=str(e),
                source_agent=self.agent_name or "unknown",
                run_id=run_id,
            )
        finally:
            # Close the client after streaming completes
            await client.aclose()

        # Emit delegation end event
        yield AgentEvent.delegation_end(
            target_agent=self.agent_name or "unknown",
            result=final_result[:500] if final_result else "Task completed",
            run_id=run_id,
            success=success,
        )

    async def _parse_sse_stream(
        self,
        response: httpx.Response,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Parse SSE stream from response.

        Args:
            response: HTTP response with SSE stream.

        Yields:
            Parsed JSON data from each SSE event.
        """
        buffer = ""

        async for chunk in response.aiter_text():
            buffer += chunk

            # Process complete events (separated by double newlines)
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                event_data = self._parse_sse_event(event_str)
                if event_data:
                    yield event_data

        # Handle any remaining data
        if buffer.strip():
            event_data = self._parse_sse_event(buffer)
            if event_data:
                yield event_data

    def _parse_sse_event(self, event_str: str) -> Optional[Dict[str, Any]]:
        """Parse a single SSE event string.

        Args:
            event_str: Raw SSE event string.

        Returns:
            Parsed JSON data or None if parsing fails.
        """
        data_line = None

        for line in event_str.strip().split("\n"):
            if line.startswith("data:"):
                data_line = line[5:].strip()
                break

        if not data_line:
            return None

        try:
            return json.loads(data_line)
        except json.JSONDecodeError as e:
            logger.debug("Failed to parse SSE data: %s", e)
            return None

    def _convert_to_agent_event(
        self,
        sse_data: Dict[str, Any],
        run_id: str,
    ) -> Optional[AgentEvent]:
        """Convert SSE data to AgentEvent.

        Parses A2A TaskStatusUpdateEvent and TaskArtifactUpdateEvent
        into AgentEvent format.

        Args:
            sse_data: Parsed SSE event data.
            run_id: Run ID for this delegation.

        Returns:
            AgentEvent or None if not convertible.
        """
        source_agent = self.agent_name or "unknown"

        # Handle JSON-RPC result wrapper
        if "result" in sse_data:
            result = sse_data["result"]
        else:
            result = sse_data

        # TaskStatusUpdateEvent
        if "state" in result:
            message = result.get("message", {})
            parts = message.get("parts", [])

            for part in parts:
                # Check for structured event data
                if "data" in part:
                    event_data = part["data"]
                    event_type = event_data.get("event_type", "")

                    # Map A2A event types to AgentEvent types
                    if event_type == "tool_start":
                        return AgentEvent.sub_agent_tool_start(
                            tool_name=event_data.get("tool_name", "unknown"),
                            source_agent=source_agent,
                            input_data=event_data.get("input", {}),
                            run_id=run_id,
                        )
                    elif event_type == "tool_end":
                        return AgentEvent.sub_agent_tool_end(
                            tool_name=event_data.get("tool_name", "unknown"),
                            source_agent=source_agent,
                            output=event_data.get("output", ""),
                            run_id=run_id,
                            duration_ms=event_data.get("duration_ms", 0),
                        )
                    elif event_type == "llm_thinking":
                        return AgentEvent(
                            type="sub_agent_thinking",
                            source_agent=source_agent,
                            run_id=run_id,
                            data=event_data,
                        )

                # Check for text content
                elif "text" in part:
                    text = part["text"]
                    # Skip emoji-prefixed status messages, pass through actual content
                    if not text.startswith(("🔧", "✅", "🤔", "🚀", "💡")):
                        logger.debug("Status message: %s", text)

            return None

        # TaskArtifactUpdateEvent
        if "artifact" in result:
            artifact = result["artifact"]
            artifact_name = artifact.get("name", "")
            parts = artifact.get("parts", [])

            for part in parts:
                if "text" in part:
                    text = part["text"]

                    # Response artifact -> text delta
                    if artifact_name == "response":
                        return AgentEvent.sub_agent_text_delta(
                            content=text,
                            source_agent=source_agent,
                            run_id=run_id,
                        )

                    # Tool call artifact
                    elif artifact_name == "tool_call":
                        try:
                            tool_data = json.loads(text)
                            return AgentEvent.sub_agent_tool_start(
                                tool_name=tool_data.get("tool", "unknown"),
                                source_agent=source_agent,
                                input_data=tool_data.get("input", {}),
                                run_id=run_id,
                            )
                        except json.JSONDecodeError:
                            pass

                    # Tool result artifact
                    elif artifact_name == "tool_result":
                        try:
                            result_data = json.loads(text)
                            return AgentEvent.sub_agent_tool_end(
                                tool_name=result_data.get("tool", "unknown"),
                                source_agent=source_agent,
                                output=result_data.get("output", text),
                                run_id=run_id,
                                duration_ms=result_data.get("duration_ms", 0),
                            )
                        except json.JSONDecodeError:
                            # Plain text result
                            return AgentEvent.sub_agent_tool_end(
                                tool_name="unknown",
                                source_agent=source_agent,
                                output=text,
                                run_id=run_id,
                            )

                    # Thinking artifact
                    elif artifact_name == "thinking":
                        return AgentEvent(
                            type="sub_agent_thinking",
                            source_agent=source_agent,
                            run_id=run_id,
                            data={"content": text},
                        )

        return None


async def create_streaming_client(
    url: str,
    name: Optional[str] = None,
    timeout: float = 120.0,
) -> StreamingA2AClient:
    """Create and connect a streaming A2A client.

    Convenience function that creates and connects a StreamingA2AClient.

    Args:
        url: Base URL of the A2A agent.
        name: Optional agent name.
        timeout: HTTP timeout in seconds.

    Returns:
        Connected StreamingA2AClient.

    Example:
        client = await create_streaming_client("http://localhost:10001")
        async for event in client.send_message_streaming("Hello"):
            print(event)
    """
    client = StreamingA2AClient(url, agent_name=name, timeout=timeout)
    await client.connect()
    return client
