"""A2A Streaming Middleware for real-time event propagation.

This middleware intercepts agent execution to emit events for:
- Agent lifecycle (before_agent, after_agent)
- LLM thinking process (before_model, wrap_model_call)
- Tool execution (wrap_tool_call) - uses this method, NOT before_tool/after_tool

Key insight:
- wrap_model_call: Intercept LLM calls, emit thinking/tool_decision events
- wrap_tool_call: Intercept tool execution, emit tool_start/tool_end events
- astream_events: Still primary event source, middleware is supplementary

Usage:
    middleware = A2AStreamingMiddleware(
        agent_name="orchestrator",
        event_queue=queue,  # Set dynamically in executor
    )
    agent = create_agent(model, tools, middleware=[
        skill_middleware,      # Tool filtering (Progressive Disclosure)
        streaming_middleware,  # Event streaming
    ])
"""

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

try:
    from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

    HAS_AGENT_MIDDLEWARE = True
except ImportError:
    HAS_AGENT_MIDDLEWARE = False
    AgentMiddleware = object
    ModelRequest = Any
    ModelResponse = Any

if TYPE_CHECKING:
    from a2a.server.events import EventQueue

logger = logging.getLogger(__name__)


class A2AStreamingMiddleware(AgentMiddleware if HAS_AGENT_MIDDLEWARE else object):
    """Middleware that emits A2A streaming events during agent execution.

    Uses LangChain v1.x available hooks:
    - before_agent / after_agent: Agent lifecycle
    - before_model / after_model: LLM call lifecycle
    - wrap_model_call: Intercept LLM calls
    - wrap_tool_call: Intercept tool execution (KEY!)

    Note: before_tool/after_tool do NOT exist in LangChain v1.x.
    Use wrap_tool_call instead.

    Attributes:
        agent_name: Name of this agent (for event metadata).
        event_queue: A2A EventQueue for streaming events.
        emit_thinking: Whether to emit thinking phase events.
        max_calls: Maximum model calls before warning.

    Example:
        middleware = A2AStreamingMiddleware(agent_name="orchestrator")
        # event_queue is set dynamically by executor before each invocation
    """

    def __init__(
        self,
        agent_name: str,
        event_queue: Optional["EventQueue"] = None,
        emit_thinking: bool = True,
        max_calls: int = 10,
    ) -> None:
        """Initialize streaming middleware.

        Args:
            agent_name: Name of this agent.
            event_queue: A2A EventQueue (optional, can be set later).
            emit_thinking: Whether to emit LLM thinking events.
            max_calls: Maximum model calls before warning.
        """
        if HAS_AGENT_MIDDLEWARE:
            super().__init__()

        self.agent_name = agent_name
        self.event_queue = event_queue
        self.emit_thinking = emit_thinking
        self.max_calls = max_calls

        # Per-invocation state (reset on before_agent)
        self._call_count = 0
        self._tool_start_times: Dict[str, float] = {}

        # A2A context for streaming events (set by executor before each invocation)
        self.context_id: Optional[str] = None
        self.task_id: Optional[str] = None

    def reset(self) -> None:
        """Reset per-invocation state."""
        self._call_count = 0
        self._tool_start_times.clear()

    # =========================================================================
    # Lifecycle Hooks
    # =========================================================================

    def before_agent(
        self,
        state: Dict[str, Any],
        runtime: Any,
    ) -> Optional[Dict[str, Any]]:
        """Called before agent execution starts.

        Args:
            state: Current agent state.
            runtime: LangGraph runtime context.

        Returns:
            State update dict or None.
        """
        self.reset()
        self._emit_status(f"🚀 {self.agent_name} started", "agent_start")
        return None

    async def abefore_agent(
        self,
        state: Dict[str, Any],
        runtime: Any,
    ) -> Optional[Dict[str, Any]]:
        """Async version of before_agent."""
        self.reset()
        self._emit_status(f"🚀 {self.agent_name} started", "agent_start")
        return None

    def after_agent(
        self,
        state: Dict[str, Any],
        runtime: Any,
    ) -> Optional[Dict[str, Any]]:
        """Called after agent execution completes.

        Args:
            state: Final agent state.
            runtime: LangGraph runtime context.

        Returns:
            State update dict or None.
        """
        self._emit_status(f"✅ {self.agent_name} completed", "agent_end")
        return None

    async def aafter_agent(
        self,
        state: Dict[str, Any],
        runtime: Any,
    ) -> Optional[Dict[str, Any]]:
        """Async version of after_agent."""
        self._emit_status(f"✅ {self.agent_name} completed", "agent_end")
        return None

    # =========================================================================
    # Model Call Hooks
    # =========================================================================

    def before_model(
        self,
        state: Dict[str, Any],
        runtime: Any,
    ) -> Optional[Dict[str, Any]]:
        """Called before each LLM invocation.

        Args:
            state: Current agent state.
            runtime: LangGraph runtime context.

        Returns:
            State update dict or None.
        """
        self._call_count += 1

        if self._call_count > self.max_calls:
            logger.warning(
                "%s exceeded max_calls (%d), consider increasing or investigating",
                self.agent_name,
                self.max_calls,
            )

        if self.emit_thinking:
            if self._call_count == 1:
                phase = "Analyzing request"
            else:
                phase = f"Thinking (round {self._call_count})"
            self._emit_status(f"🤔 {phase}...", "llm_thinking")

        return None

    async def abefore_model(
        self,
        state: Dict[str, Any],
        runtime: Any,
    ) -> Optional[Dict[str, Any]]:
        """Async version of before_model."""
        return self.before_model(state, runtime)

    def after_model(
        self,
        state: Dict[str, Any],
        runtime: Any,
    ) -> Optional[Dict[str, Any]]:
        """Called after each LLM invocation."""
        return None

    async def aafter_model(
        self,
        state: Dict[str, Any],
        runtime: Any,
    ) -> Optional[Dict[str, Any]]:
        """Async version of after_model."""
        return None

    # =========================================================================
    # Model Call Interception
    # =========================================================================

    def wrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], "ModelResponse"],
    ) -> "ModelResponse":
        """Intercept LLM call to detect tool call decisions.

        Args:
            request: Model request with messages and tools.
            handler: Next handler in chain.

        Returns:
            Model response.
        """
        response = handler(request)

        # Check if model decided to call tools
        if self._has_tool_calls(response):
            tool_names = self._extract_tool_names(response)
            self._emit_status(
                f"💡 Decided to call: {', '.join(tool_names)}",
                "tool_decision",
                {"tools": tool_names},
            )

        return response

    async def awrap_model_call(
        self,
        request: "ModelRequest",
        handler: Callable[["ModelRequest"], "ModelResponse"],
    ) -> "ModelResponse":
        """Async version of wrap_model_call."""
        response = await handler(request)

        if self._has_tool_calls(response):
            tool_names = self._extract_tool_names(response)
            self._emit_status(
                f"💡 Decided to call: {', '.join(tool_names)}",
                "tool_decision",
                {"tools": tool_names},
            )

        return response

    # =========================================================================
    # Tool Call Interception (KEY METHOD!)
    # =========================================================================

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Intercept tool execution to emit tool_start/tool_end events.

        This is the correct way to intercept tool calls in LangChain v1.x.
        before_tool/after_tool hooks do NOT exist.

        Args:
            request: Tool call request with tool_call dict.
            handler: Next handler to execute the tool.

        Returns:
            Tool result (ToolMessage or Command).
        """
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "unknown")
        tool_id = tool_call.get("id", "")
        tool_args = tool_call.get("args", {})

        # Record start time
        self._tool_start_times[tool_id] = time.time()

        # Emit tool start
        self._emit_tool_event(
            "tool_start",
            tool_name,
            tool_args,
        )

        # Execute tool
        result = handler(request)

        # Calculate duration
        start_time = self._tool_start_times.pop(tool_id, time.time())
        duration_ms = int((time.time() - start_time) * 1000)

        # Extract output for display
        output = self._extract_tool_output(result)

        # Emit tool end
        self._emit_tool_event(
            "tool_end",
            tool_name,
            tool_args,
            output,
            duration_ms,
        )

        return result

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Async version of wrap_tool_call."""
        tool_call = getattr(request, "tool_call", {}) or {}
        tool_name = tool_call.get("name", "unknown")
        tool_id = tool_call.get("id", "")
        tool_args = tool_call.get("args", {})

        self._tool_start_times[tool_id] = time.time()
        self._emit_tool_event("tool_start", tool_name, tool_args)

        result = await handler(request)

        start_time = self._tool_start_times.pop(tool_id, time.time())
        duration_ms = int((time.time() - start_time) * 1000)
        output = self._extract_tool_output(result)

        self._emit_tool_event("tool_end", tool_name, tool_args, output, duration_ms)

        return result

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _has_tool_calls(self, response: Any) -> bool:
        """Check if response contains tool calls."""
        if hasattr(response, "message") and isinstance(response.message, AIMessage):
            return bool(getattr(response.message, "tool_calls", None))
        return False

    def _extract_tool_names(self, response: Any) -> List[str]:
        """Extract tool names from response."""
        if hasattr(response, "message") and hasattr(response.message, "tool_calls"):
            tool_calls = response.message.tool_calls or []
            return [tc.get("name", "unknown") for tc in tool_calls]
        return []

    def _extract_tool_output(self, result: Any) -> str:
        """Extract output string from tool result."""
        if isinstance(result, ToolMessage):
            content = result.content
            if isinstance(content, str):
                return content[:2000]
            return str(content)[:2000]
        elif isinstance(result, Command):
            return self._extract_command_output(result)
        elif isinstance(result, str):
            return result[:2000]
        return str(result)[:2000]

    def _extract_command_output(self, command: Command) -> str:
        """Extract output from Command's ToolMessage."""
        update = getattr(command, "update", None) or {}
        messages = update.get("messages", [])

        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content
                if isinstance(content, str):
                    return content[:2000]
                return str(content)[:2000]

        return "Command executed"

    def _emit_status(
        self,
        content: str,
        event_type: str,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a status update event.

        Note: This method schedules the event emission asynchronously since
        middleware hooks are synchronous but EventQueue.enqueue_event is async.
        """
        if not self.event_queue:
            logger.debug("[%s] %s: %s", self.agent_name, event_type, content)
            return

        try:
            import asyncio
            from uuid import uuid4

            from a2a.types import Message, Part, Role, TaskState, TaskStatus, TaskStatusUpdateEvent

            data = {
                "event_type": event_type,
                "agent_name": self.agent_name,
            }
            if extra_data:
                data.update(extra_data)

            # Use context_id and task_id if available, otherwise generate UUIDs
            context_id = self.context_id or str(uuid4())
            task_id = self.task_id or str(uuid4())

            event = TaskStatusUpdateEvent(
                contextId=context_id,
                taskId=task_id,
                final=False,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        messageId=str(uuid4()),
                        role=Role.agent,
                        parts=[
                            Part(text=content),
                            Part(data=data),
                        ],
                    ),
                ),
            )

            # Schedule async enqueue if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.event_queue.enqueue_event(event))
            except RuntimeError:
                # No running loop - this shouldn't happen in normal A2A execution
                logger.debug("No event loop available for event emission")

        except Exception as e:
            logger.warning("Failed to emit status event: %s", e)

    def _emit_tool_event(
        self,
        event_type: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: str = "",
        duration_ms: int = 0,
    ) -> None:
        """Emit tool-related event.

        Note: This method schedules the event emission asynchronously since
        middleware hooks are synchronous but EventQueue.enqueue_event is async.
        """
        if not self.event_queue:
            if event_type == "tool_start":
                logger.debug("[%s] 🔧 %s starting", self.agent_name, tool_name)
            else:
                logger.debug("[%s] ✅ %s done (%dms)", self.agent_name, tool_name, duration_ms)
            return

        try:
            import asyncio
            from uuid import uuid4

            from a2a.types import Message, Part, Role, TaskState, TaskStatus, TaskStatusUpdateEvent

            if event_type == "tool_start":
                content = f"🔧 Calling: {tool_name}"
                data = {
                    "event_type": "tool_start",
                    "agent_name": self.agent_name,
                    "tool_name": tool_name,
                    "input": self._safe_serialize(tool_input),
                }
            else:  # tool_end
                content = f"✅ {tool_name} done ({duration_ms}ms)"
                data = {
                    "event_type": "tool_end",
                    "agent_name": self.agent_name,
                    "tool_name": tool_name,
                    "output": tool_output,
                    "duration_ms": duration_ms,
                }

            # Use context_id and task_id if available, otherwise generate UUIDs
            context_id = self.context_id or str(uuid4())
            task_id = self.task_id or str(uuid4())

            event = TaskStatusUpdateEvent(
                contextId=context_id,
                taskId=task_id,
                final=False,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        messageId=str(uuid4()),
                        role=Role.agent,
                        parts=[
                            Part(text=content),
                            Part(data=data),
                        ],
                    ),
                ),
            )

            # Schedule async enqueue if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.event_queue.enqueue_event(event))
            except RuntimeError:
                # No running loop - this shouldn't happen in normal A2A execution
                logger.debug("No event loop available for event emission")

        except Exception as e:
            logger.warning("Failed to emit tool event: %s", e)

    def _safe_serialize(self, obj: Any) -> Any:
        """Safely serialize object for JSON."""
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)
