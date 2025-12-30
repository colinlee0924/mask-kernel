"""MASK LLM model abstraction layer.

This module provides a tier-based model selection system that abstracts
away provider-specific details.

Usage:
    from mask.models import LLMFactory, ModelTier

    factory = LLMFactory()
    model = factory.get_model(tier=ModelTier.THINKING)
"""

from mask.models.config import (
    build_tier_mapping_from_config,
    get_api_key,
    get_default_provider_from_config,
    get_model_kwargs_from_config,
    load_model_config,
    validate_provider_config,
)
from mask.models.llm_factory import (
    DEFAULT_TIER_MAPPING,
    SUPPORTED_PROVIDERS,
    LLMFactory,
    ModelTier,
)

__all__ = [
    # Factory
    "LLMFactory",
    "ModelTier",
    "DEFAULT_TIER_MAPPING",
    "SUPPORTED_PROVIDERS",
    # Config utilities
    "load_model_config",
    "build_tier_mapping_from_config",
    "get_default_provider_from_config",
    "get_model_kwargs_from_config",
    "get_api_key",
    "validate_provider_config",
]
