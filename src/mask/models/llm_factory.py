"""LLM Factory with capability-based model selection.

This module provides a factory pattern for obtaining LLM instances based on
capability tiers rather than specific model names.

The tier system abstracts away provider-specific model names, allowing
developers to request models by capability level:
- FAST: Quick responses, lower cost (e.g., GPT-4o-mini, Claude Haiku)
- THINKING: Balanced reasoning (e.g., GPT-4o, Claude Sonnet)
- PRO: Complex reasoning (e.g., o1, Claude Opus)

Usage:
    from mask.models import LLMFactory, ModelTier

    factory = LLMFactory()

    # Get model by capability tier
    fast_model = factory.get_model(tier=ModelTier.FAST)
    thinking_model = factory.get_model(tier=ModelTier.THINKING)
    pro_model = factory.get_model(tier=ModelTier.PRO)

    # Override provider
    openai_model = factory.get_model(tier=ModelTier.THINKING, provider="openai")
"""

import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseChatModel

from mask.core.exceptions import ModelNotAvailableError, ProviderNotSupportedError

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model capability tiers.

    FAST: Quick responses with lower cost, suitable for simple tasks.
    THINKING: Balanced reasoning capability, good for most tasks.
    PRO: Advanced reasoning, best for complex analysis and planning.
    """

    FAST = "fast"
    THINKING = "thinking"
    PRO = "pro"


# Default tier-to-model mappings for each provider
DEFAULT_TIER_MAPPING: Dict[ModelTier, Dict[str, str]] = {
    ModelTier.FAST: {
        "anthropic": "claude-3-5-haiku-20241022",
        "openai": "gpt-4o-mini",
        "google": "gemini-2.0-flash-exp",
    },
    ModelTier.THINKING: {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "google": "gemini-2.0-flash-thinking-exp",
    },
    ModelTier.PRO: {
        "anthropic": "claude-opus-4-20250514",
        "openai": "o1",
        "google": "gemini-2.5-pro-preview-06-05",
    },
}

# Supported providers
SUPPORTED_PROVIDERS = {"anthropic", "openai", "google"}


class LLMFactory:
    """Factory for creating LLM instances by capability tier.

    The factory abstracts away provider-specific details, allowing developers
    to request models by capability level. This makes it easy to:
    - Switch between providers without code changes
    - Use appropriate models for different task complexities
    - Override defaults through environment variables or configuration

    Attributes:
        default_provider: The default LLM provider to use.
        tier_mapping: Mapping of tiers to model names per provider.
    """

    def __init__(
        self,
        default_provider: Optional[str] = None,
        tier_mapping: Optional[Dict[ModelTier, Dict[str, str]]] = None,
    ) -> None:
        """Initialize the LLM factory.

        Args:
            default_provider: Default provider (anthropic, openai, google).
                If not specified, checks MASK_LLM_PROVIDER env var,
                then defaults to "anthropic".
            tier_mapping: Custom tier-to-model mappings. If not provided,
                uses DEFAULT_TIER_MAPPING.
        """
        self.default_provider = (
            default_provider
            or os.environ.get("MASK_LLM_PROVIDER", "anthropic")
        )
        self.tier_mapping = tier_mapping or DEFAULT_TIER_MAPPING.copy()

        if self.default_provider not in SUPPORTED_PROVIDERS:
            logger.warning(
                "Default provider '%s' may not be supported",
                self.default_provider,
            )

    def get_model(
        self,
        tier: ModelTier = ModelTier.THINKING,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Get a model instance by capability tier.

        Args:
            tier: The capability tier (FAST, THINKING, PRO).
            provider: Override the default provider.
            **kwargs: Additional arguments passed to the model constructor
                (e.g., temperature, max_tokens).

        Returns:
            A BaseChatModel instance.

        Raises:
            ProviderNotSupportedError: If the provider is not supported.
            ModelNotAvailableError: If the model cannot be instantiated.
        """
        provider = provider or self.default_provider

        # Get model name for this tier and provider
        tier_models = self.tier_mapping.get(tier, {})
        model_name = tier_models.get(provider)

        if not model_name:
            # Check for environment variable override
            env_key = f"MASK_{tier.value.upper()}_{provider.upper()}_MODEL"
            model_name = os.environ.get(env_key)

        if not model_name:
            raise ModelNotAvailableError(
                provider,
                f"tier={tier.value}",
            )

        logger.debug(
            "Creating model: provider=%s, tier=%s, model=%s",
            provider,
            tier.value,
            model_name,
        )

        return self._create_model(provider, model_name, **kwargs)

    def _create_model(
        self,
        provider: str,
        model_name: str,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create a model instance for the given provider.

        Args:
            provider: The LLM provider name.
            model_name: The model name/ID.
            **kwargs: Additional model constructor arguments.

        Returns:
            A BaseChatModel instance.

        Raises:
            ProviderNotSupportedError: If the provider is not supported.
            ModelNotAvailableError: If the model cannot be created.
        """
        try:
            if provider == "anthropic":
                return self._create_anthropic_model(model_name, **kwargs)
            elif provider == "openai":
                return self._create_openai_model(model_name, **kwargs)
            elif provider == "google":
                return self._create_google_model(model_name, **kwargs)
            else:
                raise ProviderNotSupportedError(provider)
        except ImportError as e:
            raise ModelNotAvailableError(
                provider,
                f"{model_name} (missing dependency: {e})",
            ) from e
        except Exception as e:
            raise ModelNotAvailableError(
                provider,
                f"{model_name} ({e})",
            ) from e

    def _create_anthropic_model(
        self,
        model_name: str,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create an Anthropic model instance."""
        from langchain_anthropic import ChatAnthropic

        # Set sensible defaults
        defaults = {
            "model": model_name,
        }
        defaults.update(kwargs)

        return ChatAnthropic(**defaults)

    def _create_openai_model(
        self,
        model_name: str,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create an OpenAI model instance."""
        from langchain_openai import ChatOpenAI

        defaults = {
            "model": model_name,
        }
        defaults.update(kwargs)

        return ChatOpenAI(**defaults)

    def _create_google_model(
        self,
        model_name: str,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create a Google model instance."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        defaults = {
            "model": model_name,
        }
        defaults.update(kwargs)

        return ChatGoogleGenerativeAI(**defaults)

    def set_tier_model(
        self,
        tier: ModelTier,
        provider: str,
        model_name: str,
    ) -> None:
        """Override the model for a specific tier and provider.

        Args:
            tier: The capability tier.
            provider: The provider name.
            model_name: The model name to use.
        """
        if tier not in self.tier_mapping:
            self.tier_mapping[tier] = {}
        self.tier_mapping[tier][provider] = model_name
        logger.debug(
            "Set tier model: %s/%s = %s",
            tier.value,
            provider,
            model_name,
        )

    def get_model_name(
        self,
        tier: ModelTier,
        provider: Optional[str] = None,
    ) -> Optional[str]:
        """Get the model name for a tier and provider without instantiating.

        Args:
            tier: The capability tier.
            provider: The provider (defaults to default_provider).

        Returns:
            The model name or None if not configured.
        """
        provider = provider or self.default_provider
        return self.tier_mapping.get(tier, {}).get(provider)

    @classmethod
    def from_env(cls) -> "LLMFactory":
        """Create a factory configured from environment variables.

        Environment variables:
        - MASK_LLM_PROVIDER: Default provider
        - MASK_FAST_ANTHROPIC_MODEL: Override fast tier Anthropic model
        - MASK_THINKING_OPENAI_MODEL: Override thinking tier OpenAI model
        - etc.

        Returns:
            Configured LLMFactory instance.
        """
        factory = cls()

        # Check for model overrides in environment
        for tier in ModelTier:
            for provider in SUPPORTED_PROVIDERS:
                env_key = f"MASK_{tier.value.upper()}_{provider.upper()}_MODEL"
                model_name = os.environ.get(env_key)
                if model_name:
                    factory.set_tier_model(tier, provider, model_name)
                    logger.debug("Loaded model override from %s", env_key)

        return factory
