"""Base Agent implementation.

This module provides the BaseAgent abstract class that serves as the
foundation for MASK agents. It supports both stateless and stateful
operation modes.

Key features:
- Stateless by default (stateless=True)
- Optional session persistence for stateful operation
- Integration with SkillRegistry for Progressive Disclosure
- LangGraph-based execution with create_react_agent
- Built-in observability via Langfuse callbacks with proper trace hierarchy
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from mask.core.registry import SkillRegistry
from mask.core.state import SkillState
from mask.middleware.skill_middleware import SkillMiddleware
from mask.session.session import Session
from mask.storage.base import SessionStore
from mask.storage.memory_store import MemorySessionStore

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for MASK agents.

    Provides a foundation for building agents with:
    - Progressive Disclosure of skills
    - Optional session persistence
    - LangGraph-based execution

    By default, agents are stateless (stateless=True). Set stateless=False
    and provide a SessionStore for stateful operation.

    Example:
        class MyAgent(BaseAgent):
            async def invoke(self, message, session_id=None):
                # Implementation
                pass

        agent = MyAgent(
            model=model,
            skill_registry=registry,
            system_prompt="You are a helpful assistant.",
        )
    """

    def __init__(
        self,
        model: BaseChatModel,
        skill_registry: Optional[SkillRegistry] = None,
        system_prompt: str = "You are a helpful assistant.",
        *,
        name: Optional[str] = None,
        stateless: bool = True,
        session_store: Optional[SessionStore] = None,
        additional_tools: Optional[List[BaseTool]] = None,
        middleware: Optional[SkillMiddleware] = None,
        enable_observability: bool = True,
    ) -> None:
        """Initialize the base agent.

        Args:
            model: The LLM model to use.
            skill_registry: Optional skill registry for Progressive Disclosure.
            system_prompt: The system prompt for the agent.
            name: Agent name for observability traces. If not provided, uses class name.
            stateless: If True (default), no session state is maintained.
            session_store: Storage backend for sessions. Required for stateful.
            additional_tools: Non-skill tools to always include.
            middleware: Custom skill middleware. Created automatically if not provided.
            enable_observability: If True (default), auto-detect and use Langfuse tracing.
        """
        self.model = model
        self.skill_registry = skill_registry or SkillRegistry()
        self.system_prompt = system_prompt
        self.name = name or self.__class__.__name__
        self.stateless = stateless
        self.additional_tools = additional_tools or []
        self.enable_observability = enable_observability

        # Session store setup
        if not stateless:
            self._session_store = session_store or MemorySessionStore()
        else:
            self._session_store = None

        # Middleware setup
        self.middleware = middleware or SkillMiddleware(self.skill_registry)

        logger.debug(
            "Initialized BaseAgent: name=%s, stateless=%s, skills=%d, observability=%s",
            self.name,
            stateless,
            len(self.skill_registry),
            enable_observability,
        )

    def _get_callbacks(self, session_id: Optional[str] = None) -> List[Any]:
        """Get callback handlers for model invocation.

        Args:
            session_id: Optional session ID for trace grouping.

        Returns:
            List of callback handlers (e.g., Langfuse handler).
        """
        if not self.enable_observability:
            return []

        callbacks: List[Any] = []
        try:
            from mask.observability import get_langfuse_handler

            handler = get_langfuse_handler(
                trace_name=self.name,
                session_id=session_id,
            )
            if handler:
                callbacks.append(handler)
                logger.debug("Added Langfuse callback handler for agent=%s", self.name)
        except ImportError:
            pass

        return callbacks

    @abstractmethod
    async def invoke(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Process a single message and return the response.

        Args:
            message: The user message to process.
            session_id: Optional session ID for stateful operation.

        Returns:
            The agent's response as a string.
        """
        pass

    @abstractmethod
    async def stream(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Process a message and stream the response.

        Args:
            message: The user message to process.
            session_id: Optional session ID for stateful operation.

        Yields:
            Response chunks as strings.
        """
        pass

    async def _get_session(self, session_id: Optional[str]) -> Optional[Session]:
        """Get or create a session for stateful operation.

        Args:
            session_id: The session ID.

        Returns:
            Session object or None if stateless.
        """
        if self.stateless or session_id is None or self._session_store is None:
            return None

        return await self._session_store.get_or_create(session_id)

    async def _save_session(self, session: Optional[Session]) -> None:
        """Save session state.

        Args:
            session: The session to save.
        """
        if session is not None and self._session_store is not None:
            await self._session_store.save(session)

    def _build_state(
        self,
        messages: Sequence[BaseMessage],
        skills_loaded: Optional[List[str]] = None,
    ) -> SkillState:
        """Build agent state from messages and skills.

        Args:
            messages: Conversation messages.
            skills_loaded: List of active skill names.

        Returns:
            SkillState dictionary.
        """
        return {
            "messages": list(messages),
            "skills_loaded": skills_loaded or [],
        }

    def _get_tools(self, state: SkillState) -> List[BaseTool]:
        """Get tools for current state.

        Args:
            state: Current agent state.

        Returns:
            List of available tools.
        """
        return self.middleware.get_tools(state, self.additional_tools)

    def _prepare_messages(
        self,
        state: SkillState,
        include_system: bool = True,
    ) -> List[BaseMessage]:
        """Prepare messages with skill information.

        Args:
            state: Current agent state.
            include_system: Whether to include system prompt.

        Returns:
            Prepared message list.
        """
        if include_system:
            return self.middleware.prepare_messages(state)
        return list(state.get("messages", []))


def _default_agent_factory(model, tools, system_prompt):
    """Default agent factory using LangChain v1.x create_agent.

    Args:
        model: The LLM model.
        tools: List of tools for the agent.
        system_prompt: System prompt for the agent.

    Returns:
        Agent instance (LangGraph-based).
    """
    from langchain.agents import create_agent

    return create_agent(model, tools=tools, system_prompt=system_prompt)


class SimpleAgent(BaseAgent):
    """Simple agent implementation with pluggable agent factory.

    By default uses LangChain v1.x create_agent. You can pass a custom
    agent_factory to use different agent implementations (e.g., deepagents).

    Observability (Phoenix/Langfuse) is handled uniformly by mask-kernel.
    Developers only need to configure .env (project, api-key, baseurl).

    Example:
        # Default: LangChain create_agent
        agent = SimpleAgent(
            model=factory.get_model(tier=ModelTier.THINKING),
            name="my-agent",
            system_prompt="You are a helpful assistant.",
        )

        # Custom: Using deepagents
        from deepagents import create_deep_agent
        agent = SimpleAgent(
            model=factory.get_model(tier=ModelTier.THINKING),
            agent_factory=create_deep_agent,
            ...
        )
    """

    def __init__(
        self,
        *args: Any,
        agent_factory: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SimpleAgent.

        Args:
            agent_factory: Custom agent factory function. Signature:
                (model, tools, system_prompt) -> Agent
                Defaults to LangChain v1.x create_agent.
        """
        super().__init__(*args, **kwargs)
        self._graph = None  # Lazy-initialized agent
        self.agent_factory = agent_factory or _default_agent_factory

    def _get_graph(self, tools: List[BaseTool]) -> Any:
        """Create agent using the configured factory.

        Args:
            tools: List of tools for the agent.

        Returns:
            Agent instance.
        """
        return self.agent_factory(self.model, tools, self.system_prompt)

    async def invoke(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Process a message and return the response.

        Args:
            message: The user message.
            session_id: Optional session ID.

        Returns:
            The agent's response.
        """
        # Get or create session
        session = await self._get_session(session_id)

        # Build messages
        messages: List[BaseMessage] = []
        if session:
            messages = list(session.messages)
        messages.append(HumanMessage(content=message))

        # Build state
        skills_loaded = session.skills_loaded if session else []
        state = self._build_state(messages, skills_loaded)

        # Get tools
        tools = self._get_tools(state)

        # Get callbacks for observability with session_id
        callbacks = self._get_callbacks(session_id=session_id)
        config: Dict[str, Any] = {
            "callbacks": callbacks,
            "run_name": self.name,  # Set trace name for observability
        }

        # Add session/thread configuration for LangGraph
        if session_id:
            config["configurable"] = {"thread_id": session_id}
            # Set session.id for Phoenix/OpenInference session tracking
            config["metadata"] = {"session.id": session_id}

        # Always use LangGraph agent (create_agent) for proper trace structure
        # Even with empty tools, this ensures Phoenix/Langfuse see full execution details:
        # LangGraph → model → ChatAnthropic (like basic-observability-examples)
        graph = self._get_graph(tools)  # tools can be empty list

        # Use OpenInference session context if available for Phoenix tracking
        result = await self._invoke_with_session_context(
            graph, messages, config, session_id
        )

        # Extract last message from result
        result_messages = result.get("messages", [])
        if result_messages:
            last_message = result_messages[-1]
            response_content = self._extract_content(last_message)
        else:
            response_content = ""

        # Update session
        if session:
            session.add_message(HumanMessage(content=message))
            session.add_message(AIMessage(content=response_content))
            await self._save_session(session)

        return response_content

    async def _invoke_with_session_context(
        self,
        graph: Any,
        messages: List[BaseMessage],
        config: Dict[str, Any],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        """Invoke graph with optional session context for Phoenix tracking."""
        if session_id:
            try:
                from openinference.instrumentation import using_session

                with using_session(session_id):
                    return await graph.ainvoke({"messages": messages}, config=config)
            except ImportError:
                pass
        return await graph.ainvoke({"messages": messages}, config=config)

    async def stream(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream the response.

        Args:
            message: The user message.
            session_id: Optional session ID.

        Yields:
            Response chunks.
        """
        # Get or create session
        session = await self._get_session(session_id)

        # Build messages
        messages: List[BaseMessage] = []
        if session:
            messages = list(session.messages)
        messages.append(HumanMessage(content=message))

        # Build state
        skills_loaded = session.skills_loaded if session else []
        state = self._build_state(messages, skills_loaded)

        # Get tools
        tools = self._get_tools(state)

        # Get callbacks for observability with session_id
        callbacks = self._get_callbacks(session_id=session_id)
        config: Dict[str, Any] = {
            "callbacks": callbacks,
            "run_name": self.name,  # Set trace name for observability
        }

        # Add session/thread configuration for LangGraph
        if session_id:
            config["configurable"] = {"thread_id": session_id}
            # Set session.id for Phoenix/OpenInference session tracking
            config["metadata"] = {"session.id": session_id}

        # Always use LangGraph agent (create_agent) for proper trace structure
        full_response = ""
        graph = self._get_graph(tools)  # tools can be empty list

        # Use OpenInference session context if available for Phoenix tracking
        async for chunk in self._stream_with_session_context(
            graph, messages, config, session_id
        ):
            full_response += chunk
            yield chunk

        # Update session
        if session:
            session.add_message(HumanMessage(content=message))
            session.add_message(AIMessage(content=full_response))
            await self._save_session(session)

    async def _stream_with_session_context(
        self,
        graph: Any,
        messages: List[BaseMessage],
        config: Dict[str, Any],
        session_id: Optional[str],
    ) -> AsyncIterator[str]:
        """Stream graph with optional session context for Phoenix tracking."""
        session_context = None
        if session_id:
            try:
                from openinference.instrumentation import using_session

                session_context = using_session(session_id)
                session_context.__enter__()
            except ImportError:
                session_context = None

        try:
            async for event in graph.astream(
                {"messages": messages},
                config=config,
                stream_mode="messages",
            ):
                # Extract content from streaming events
                if isinstance(event, tuple) and len(event) == 2:
                    msg, metadata = event
                    if hasattr(msg, "content") and msg.content:
                        content = msg.content
                        if isinstance(content, str):
                            yield content
        finally:
            if session_context:
                session_context.__exit__(None, None, None)

    def _extract_content(self, response: Any) -> str:
        """Extract text content from model response.

        Args:
            response: Model response object.

        Returns:
            Response content as string.
        """
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Handle content blocks
                return "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
        return str(response)
