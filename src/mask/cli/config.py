"""CLI Configuration for MASK Kernel.

This module provides a configuration-based approach for the CLI,
similar to deepagents. It handles:
- Environment variable loading
- Agent directory management
- Model detection and creation
- Settings persistence

Configuration directories:
- User-level: ~/.mask/<agent_name>/
- Project-level: .mask/
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Default directories
DEFAULT_USER_CONFIG_DIR = Path.home() / ".mask"
DEFAULT_AGENT_NAME = "agent"

# Valid agent name pattern (alphanumeric, hyphens, underscores)
AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]*$")


def validate_agent_name(name: str) -> bool:
    """Validate agent name format.

    Agent names must:
    - Start with a letter
    - Contain only alphanumeric characters, hyphens, and underscores
    - Be at most 64 characters

    Args:
        name: The agent name to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not name or len(name) > 64:
        return False
    return bool(AGENT_NAME_PATTERN.match(name))


def find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """Find the project root by looking for .git directory.

    Args:
        start_path: Starting directory. Defaults to cwd.

    Returns:
        Project root path or None if not found.
    """
    current = Path(start_path or os.getcwd()).resolve()

    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent

    return None


@dataclass
class Settings:
    """CLI settings and configuration.

    Centralizes all configuration including API keys, paths, and model settings.
    """

    # API Keys (from environment)
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    google_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY")
    )

    # Observability
    phoenix_endpoint: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
        )
    )
    langfuse_public_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("LANGFUSE_PUBLIC_KEY")
    )
    langfuse_secret_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("LANGFUSE_SECRET_KEY")
    )

    # Model settings
    default_provider: str = field(
        default_factory=lambda: os.environ.get("MASK_LLM_PROVIDER", "anthropic")
    )
    default_model: Optional[str] = field(
        default_factory=lambda: os.environ.get("MASK_DEFAULT_MODEL")
    )

    # Directories
    user_config_dir: Path = field(default_factory=lambda: DEFAULT_USER_CONFIG_DIR)
    project_root: Optional[Path] = field(default_factory=find_project_root)

    # Agent
    agent_name: str = DEFAULT_AGENT_NAME

    def __post_init__(self):
        """Ensure directories exist."""
        self.user_config_dir.mkdir(parents=True, exist_ok=True)

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key)

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key)

    @property
    def has_google(self) -> bool:
        """Check if Google API key is configured."""
        return bool(self.google_api_key)

    def get_agent_dir(self, agent_name: Optional[str] = None) -> Path:
        """Get the agent configuration directory.

        Args:
            agent_name: Agent name. Uses default if not provided.

        Returns:
            Path to agent directory.
        """
        name = agent_name or self.agent_name
        return self.user_config_dir / name

    def get_agent_skills_dir(self, agent_name: Optional[str] = None) -> Path:
        """Get the agent skills directory.

        Args:
            agent_name: Agent name. Uses default if not provided.

        Returns:
            Path to agent skills directory.
        """
        return self.get_agent_dir(agent_name) / "skills"

    def get_project_skills_dir(self) -> Optional[Path]:
        """Get the project-level skills directory.

        Returns:
            Path to project skills directory or None if not in a project.
        """
        if self.project_root:
            return self.project_root / ".mask" / "skills"
        return None

    def get_agent_config_path(self, agent_name: Optional[str] = None) -> Path:
        """Get the agent.md configuration file path.

        Args:
            agent_name: Agent name. Uses default if not provided.

        Returns:
            Path to agent.md file.
        """
        return self.get_agent_dir(agent_name) / "agent.md"

    def ensure_agent_dir(self, agent_name: Optional[str] = None) -> Path:
        """Ensure agent directory exists and create if needed.

        Args:
            agent_name: Agent name. Uses default if not provided.

        Returns:
            Path to agent directory.
        """
        agent_dir = self.get_agent_dir(agent_name)
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Create skills directory
        skills_dir = agent_dir / "skills"
        skills_dir.mkdir(exist_ok=True)

        # Create default agent.md if it doesn't exist
        agent_md = agent_dir / "agent.md"
        if not agent_md.exists():
            agent_md.write_text(
                f"# {agent_name or self.agent_name}\n\n"
                "You are a helpful AI assistant.\n",
                encoding="utf-8",
            )

        return agent_dir

    def list_agents(self) -> List[str]:
        """List all configured agents.

        Returns:
            List of agent names.
        """
        if not self.user_config_dir.exists():
            return []

        agents = []
        for path in self.user_config_dir.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                # Check if it has an agent.md file
                if (path / "agent.md").exists():
                    agents.append(path.name)

        return sorted(agents)

    def detect_provider_from_model(self, model_name: str) -> str:
        """Auto-detect provider from model name.

        Args:
            model_name: Model identifier string.

        Returns:
            Provider name ('anthropic', 'openai', 'google').
        """
        model_lower = model_name.lower()

        if model_lower.startswith("claude") or "anthropic" in model_lower:
            return "anthropic"
        elif model_lower.startswith("gpt") or model_lower.startswith("o1"):
            return "openai"
        elif model_lower.startswith("gemini") or "google" in model_lower:
            return "google"

        # Fallback to default or based on available keys
        if self.has_anthropic:
            return "anthropic"
        elif self.has_openai:
            return "openai"
        elif self.has_google:
            return "google"

        return self.default_provider

    def create_model(
        self,
        model_name: Optional[str] = None,
        tier: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        """Create an LLM model from configuration.

        Args:
            model_name: Specific model name (e.g., "claude-sonnet-4-20250514").
            tier: Model tier (fast, thinking, pro).
            provider: Provider override.

        Returns:
            Configured LLM model.
        """
        from mask.models import LLMFactory, ModelTier

        factory = LLMFactory()

        # If model_name is specified, detect provider and create directly
        if model_name:
            detected_provider = provider or self.detect_provider_from_model(model_name)

            # Map provider to factory method
            if detected_provider == "anthropic":
                return factory._create_anthropic(model_name)
            elif detected_provider == "openai":
                return factory._create_openai(model_name)
            elif detected_provider == "google":
                return factory._create_google(model_name)

        # Otherwise use tier-based selection
        tier_map = {
            "fast": ModelTier.FAST,
            "thinking": ModelTier.THINKING,
            "pro": ModelTier.PRO,
        }
        model_tier = tier_map.get(tier or "thinking", ModelTier.THINKING)

        return factory.get_model(tier=model_tier, provider=provider)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary.

        Returns:
            Settings as dictionary (excluding sensitive data).
        """
        return {
            "agent_name": self.agent_name,
            "default_provider": self.default_provider,
            "user_config_dir": str(self.user_config_dir),
            "project_root": str(self.project_root) if self.project_root else None,
            "has_anthropic": self.has_anthropic,
            "has_openai": self.has_openai,
            "has_google": self.has_google,
        }


@dataclass
class SessionState:
    """Mutable session state for CLI execution.

    Tracks runtime state that may change during a session.
    """

    # Thread/session ID
    thread_id: str = field(default_factory=lambda: uuid4().hex)

    # Auto-approval mode
    auto_approve: bool = False

    # Whether splash screen has been shown
    splash_shown: bool = False

    # Current conversation context
    context_id: Optional[str] = None

    def reset_thread(self) -> None:
        """Reset to a new thread ID."""
        self.thread_id = uuid4().hex
        self.context_id = None


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(agent_name: Optional[str] = None) -> Settings:
    """Get or create the global settings instance.

    Args:
        agent_name: Optional agent name to use.

    Returns:
        Settings instance.
    """
    global _settings

    if _settings is None:
        _settings = Settings()

    if agent_name:
        _settings.agent_name = agent_name

    return _settings


def reset_settings() -> None:
    """Reset the global settings instance."""
    global _settings
    _settings = None
