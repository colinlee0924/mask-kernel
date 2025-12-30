"""Model configuration utilities.

This module provides utilities for loading and managing model configurations
from files and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from mask.models.llm_factory import DEFAULT_TIER_MAPPING, ModelTier


def load_model_config(config_path: str | Path) -> Dict[str, Any]:
    """Load model configuration from a YAML or JSON file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the file format is not supported.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    content = config_path.read_text(encoding="utf-8")

    if config_path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(content) or {}
    elif config_path.suffix == ".json":
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def build_tier_mapping_from_config(
    config: Dict[str, Any],
) -> Dict[ModelTier, Dict[str, str]]:
    """Build tier mapping from configuration dictionary.

    Expected config format:
    ```yaml
    models:
      fast:
        anthropic: claude-3-5-haiku-20241022
        openai: gpt-4o-mini
      thinking:
        anthropic: claude-sonnet-4-20250514
      pro:
        anthropic: claude-opus-4-20250514
    ```

    Args:
        config: Configuration dictionary.

    Returns:
        Tier mapping dictionary.
    """
    # Start with defaults
    mapping = {tier: dict(models) for tier, models in DEFAULT_TIER_MAPPING.items()}

    # Override from config
    models_config = config.get("models", {})

    for tier_name, providers in models_config.items():
        try:
            tier = ModelTier(tier_name)
        except ValueError:
            continue

        if tier not in mapping:
            mapping[tier] = {}

        if isinstance(providers, dict):
            mapping[tier].update(providers)

    return mapping


def get_default_provider_from_config(
    config: Dict[str, Any],
) -> Optional[str]:
    """Get default provider from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Default provider name or None.
    """
    return config.get("default_provider")


def get_model_kwargs_from_config(
    config: Dict[str, Any],
    tier: Optional[ModelTier] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Get model constructor kwargs from configuration.

    Expected config format:
    ```yaml
    model_kwargs:
      temperature: 0.7
      max_tokens: 4096

    # Or per-tier/provider overrides
    tier_kwargs:
      pro:
        max_tokens: 8192
    ```

    Args:
        config: Configuration dictionary.
        tier: Optional tier for tier-specific kwargs.
        provider: Optional provider for provider-specific kwargs.

    Returns:
        Model kwargs dictionary.
    """
    kwargs: Dict[str, Any] = {}

    # Global kwargs
    kwargs.update(config.get("model_kwargs", {}))

    # Tier-specific kwargs
    if tier:
        tier_kwargs = config.get("tier_kwargs", {}).get(tier.value, {})
        kwargs.update(tier_kwargs)

    # Provider-specific kwargs
    if provider:
        provider_kwargs = config.get("provider_kwargs", {}).get(provider, {})
        kwargs.update(provider_kwargs)

    return kwargs


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider from environment.

    Checks standard environment variable names for each provider.

    Args:
        provider: Provider name.

    Returns:
        API key or None if not found.
    """
    env_var_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    env_var = env_var_map.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return None


def validate_provider_config(provider: str) -> bool:
    """Check if a provider is properly configured.

    Args:
        provider: Provider name.

    Returns:
        True if the provider has necessary configuration.
    """
    api_key = get_api_key(provider)
    return api_key is not None and len(api_key) > 0
