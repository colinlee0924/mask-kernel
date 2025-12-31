"""Base Agent implementation.

This module provides the BaseAgent abstract class that serves as the
foundation for MASK agents. It supports both stateless and stateful
operation modes.

Key features:
- Stateless by default (stateless=True)
- Optional session persistence for stateful operation
- Integration with SkillRegistry for Progressive Disclosure
- LangGraph-based execution
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool

from mask.core.registry import SkillRegistry
from mask.core.state import SkillState
from mask.middleware.skill_middleware import SkillMiddleware
from mask.session.session import Session
from mask.storage.base import SessionStore
from mask.storage.memory_store import MemorySessionStore

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
        stateless: bool = True,
        session_store: Optional[SessionStore] = None,
        additional_tools: Optional[List[BaseTool]] = None,
        middleware: Optional[SkillMiddleware] = None,
    ) -> None:
        """Initialize the base agent.

        Args:
            model: The LLM model to use.
            skill_registry: Optional skill registry for Progressive Disclosure.
            system_prompt: The system prompt for the agent.
            stateless: If True (default), no session state is maintained.
            session_store: Storage backend for sessions. Required for stateful.
            additional_tools: Non-skill tools to always include.
            middleware: Custom skill middleware. Created automatically if not provided.
        """
        self.model = model
        self.skill_registry = skill_registry or SkillRegistry()
        self.system_prompt = system_prompt
        self.stateless = stateless
        self.additional_tools = additional_tools or []

        # Session store setup
        if not stateless:
            self._session_store = session_store or MemorySessionStore()
        else:
            self._session_store = None

        # Middleware setup
        self.middleware = middleware or SkillMiddleware(self.skill_registry)

        logger.debug(
            "Initialized BaseAgent: stateless=%s, skills=%d",
            stateless,
            len(self.skill_registry),
        )

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


class SimpleAgent(BaseAgent):
    """Simple agent implementation using direct model calls.

    This is a basic implementation suitable for simple use cases.
    For more complex scenarios, use LangGraph-based agents.

    Example:
        agent = SimpleAgent(
            model=factory.get_model(tier=ModelTier.THINKING),
            system_prompt="You are a helpful assistant.",
        )
        response = await agent.invoke("Hello!")
    """

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

        # Get tools and prepare messages
        tools = self._get_tools(state)
        prepared_messages = self._prepare_messages(state)

        # Invoke model
        if tools:
            model_with_tools = self.model.bind_tools(tools)
            response = await model_with_tools.ainvoke(prepared_messages)
        else:
            response = await self.model.ainvoke(prepared_messages)

        # Extract response content
        response_content = self._extract_content(response)

        # Update session
        if session:
            session.add_message(HumanMessage(content=message))
            session.add_message(AIMessage(content=response_content))
            await self._save_session(session)

        return response_content

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

        # Get tools and prepare messages
        tools = self._get_tools(state)
        prepared_messages = self._prepare_messages(state)

        # Stream from model
        full_response = ""
        if tools:
            model_with_tools = self.model.bind_tools(tools)
            async for chunk in model_with_tools.astream(prepared_messages):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield chunk.content
        else:
            async for chunk in self.model.astream(prepared_messages):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield chunk.content

        # Update session
        if session:
            session.add_message(HumanMessage(content=message))
            session.add_message(AIMessage(content=full_response))
            await self._save_session(session)

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
