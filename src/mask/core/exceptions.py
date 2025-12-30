"""MASK Kernel exceptions.

This module defines all custom exceptions used throughout the MASK framework.
"""


class MaskError(Exception):
    """Base exception for all MASK errors."""

    pass


class SkillError(MaskError):
    """Base exception for skill-related errors."""

    pass


class SkillNotFoundError(SkillError):
    """Raised when a skill cannot be found."""

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        super().__init__(f"Skill not found: {skill_name}")


class SkillLoadError(SkillError):
    """Raised when a skill fails to load."""

    def __init__(self, skill_name: str, reason: str) -> None:
        self.skill_name = skill_name
        self.reason = reason
        super().__init__(f"Failed to load skill '{skill_name}': {reason}")


class SkillAlreadyRegisteredError(SkillError):
    """Raised when attempting to register a skill that already exists."""

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        super().__init__(f"Skill already registered: {skill_name}")


class SkillMetadataError(SkillError):
    """Raised when skill metadata is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Invalid skill metadata: {message}")


class SessionError(MaskError):
    """Base exception for session-related errors."""

    pass


class SessionNotFoundError(SessionError):
    """Raised when a session cannot be found."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class SessionExpiredError(SessionError):
    """Raised when a session has expired."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session expired: {session_id}")


class StorageError(MaskError):
    """Base exception for storage-related errors."""

    pass


class StorageConnectionError(StorageError):
    """Raised when storage connection fails."""

    def __init__(self, backend: str, reason: str) -> None:
        self.backend = backend
        self.reason = reason
        super().__init__(f"Failed to connect to {backend} storage: {reason}")


class ModelError(MaskError):
    """Base exception for model-related errors."""

    pass


class ModelNotAvailableError(ModelError):
    """Raised when a model is not available."""

    def __init__(self, provider: str, model_name: str) -> None:
        self.provider = provider
        self.model_name = model_name
        super().__init__(f"Model not available: {provider}/{model_name}")


class ProviderNotSupportedError(ModelError):
    """Raised when a provider is not supported."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(f"Provider not supported: {provider}")


class A2AError(MaskError):
    """Base exception for A2A-related errors."""

    pass


class AgentConnectionError(A2AError):
    """Raised when connection to remote agent fails."""

    def __init__(self, agent_url: str, reason: str) -> None:
        self.agent_url = agent_url
        self.reason = reason
        super().__init__(f"Failed to connect to agent at {agent_url}: {reason}")


class AgentNotFoundError(A2AError):
    """Raised when a remote agent cannot be found."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        super().__init__(f"Agent not found: {agent_name}")


class MCPError(MaskError):
    """Base exception for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Raised when connection to MCP server fails."""

    def __init__(self, server_name: str, reason: str) -> None:
        self.server_name = server_name
        self.reason = reason
        super().__init__(f"Failed to connect to MCP server '{server_name}': {reason}")


class MCPConfigError(MCPError):
    """Raised when MCP configuration is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Invalid MCP configuration: {message}")


class PromptError(MaskError):
    """Base exception for prompt-related errors."""

    pass


class PromptNotFoundError(PromptError):
    """Raised when a prompt file cannot be found."""

    def __init__(self, prompt_name: str, path: str) -> None:
        self.prompt_name = prompt_name
        self.path = path
        super().__init__(f"Prompt '{prompt_name}' not found at: {path}")
