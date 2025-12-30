"""Session data model.

This module defines the Session dataclass that represents a conversation
session with an agent.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


@dataclass
class Session:
    """Represents a conversation session with an agent.

    Sessions track conversation history, active skills, and metadata
    for multi-turn interactions.

    Attributes:
        session_id: Unique identifier for this session.
        user_id: Optional user identifier.
        data: Arbitrary session data dictionary.
        created_at: Session creation timestamp.
        updated_at: Last update timestamp.
        expires_at: Optional expiration timestamp.
        messages: Conversation message history.
        skills_loaded: List of currently active skill names.
        pagination_cursor: Cursor for paginated responses.
    """

    session_id: str = field(default_factory=generate_session_id)
    user_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    messages: List[BaseMessage] = field(default_factory=list)
    skills_loaded: List[str] = field(default_factory=list)
    pagination_cursor: Optional[int] = None

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()

    def is_expired(self) -> bool:
        """Check if the session has expired.

        Returns:
            True if the session has expired, False otherwise.
        """
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def set_ttl(self, seconds: int) -> None:
        """Set session expiration time from now.

        Args:
            seconds: Number of seconds until expiration.
        """
        self.expires_at = datetime.now() + timedelta(seconds=seconds)

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the conversation history.

        Args:
            message: The message to add.
        """
        self.messages.append(message)
        self.touch()

    def get_messages(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[BaseMessage]:
        """Get messages with optional pagination.

        Args:
            limit: Maximum number of messages to return.
            offset: Number of messages to skip.

        Returns:
            List of messages.
        """
        messages = self.messages[offset:]
        if limit is not None:
            messages = messages[:limit]
        return messages

    def clear_messages(self) -> None:
        """Clear all messages from the session."""
        self.messages = []
        self.touch()

    def activate_skill(self, skill_name: str) -> None:
        """Mark a skill as active.

        Args:
            skill_name: Name of the skill to activate.
        """
        if skill_name not in self.skills_loaded:
            self.skills_loaded.append(skill_name)
            self.touch()

    def deactivate_skill(self, skill_name: str) -> None:
        """Mark a skill as inactive.

        Args:
            skill_name: Name of the skill to deactivate.
        """
        if skill_name in self.skills_loaded:
            self.skills_loaded.remove(skill_name)
            self.touch()

    def set_data(self, key: str, value: Any) -> None:
        """Set a custom data value.

        Args:
            key: Data key.
            value: Data value.
        """
        self.data[key] = value
        self.touch()

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get a custom data value.

        Args:
            key: Data key.
            default: Default value if key not found.

        Returns:
            The data value or default.
        """
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization.

        Note: Messages are converted to their dict representation.

        Returns:
            Dictionary representation of the session.
        """
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "messages": [
                {
                    "type": msg.type,
                    "content": msg.content,
                }
                for msg in self.messages
            ],
            "skills_loaded": self.skills_loaded,
            "pagination_cursor": self.pagination_cursor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create a Session from dictionary.

        Note: Messages are reconstructed as basic message types.

        Args:
            data: Dictionary representation.

        Returns:
            Session instance.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        # Parse timestamps
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data["updated_at"])
        expires_at = (
            datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None
        )

        # Reconstruct messages
        messages: List[BaseMessage] = []
        for msg_data in data.get("messages", []):
            msg_type = msg_data.get("type", "human")
            content = msg_data.get("content", "")

            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
            else:
                # Default to HumanMessage for unknown types
                messages.append(HumanMessage(content=content))

        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            data=data.get("data", {}),
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
            messages=messages,
            skills_loaded=data.get("skills_loaded", []),
            pagination_cursor=data.get("pagination_cursor"),
        )
