"""Observability setup using OpenInference.

This module provides utilities for setting up observability and tracing
using OpenInference with LangChain instrumentation.

Requires: pip install mask-kernel[observability]
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def setup_openinference_tracing(
    project_name: str = "mask-agent",
    endpoint: Optional[str] = None,
    batch: bool = True,
) -> bool:
    """Set up OpenInference tracing with Arize Phoenix.

    This configures automatic tracing for LangChain operations,
    sending traces to a Phoenix server.

    Args:
        project_name: Name of the project for trace grouping.
        endpoint: Phoenix endpoint URL. If not provided, uses
            PHOENIX_COLLECTOR_ENDPOINT env var or defaults to local.
        batch: Whether to batch trace exports (recommended for production).

    Returns:
        True if setup was successful, False otherwise.

    Example:
        from mask.observability import setup_openinference_tracing

        # Setup tracing before creating agents
        setup_openinference_tracing("my-jira-agent")

        # Now all LangChain operations will be traced
        agent = create_mask_agent()
    """
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError:
        logger.warning(
            "Observability requires additional packages. "
            "Install with: pip install mask-kernel[observability]"
        )
        return False

    # Determine endpoint
    if endpoint is None:
        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")

    try:
        # Register tracer provider
        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,
            batch=batch,
        )

        # Instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        logger.info(
            "OpenInference tracing enabled: project=%s, endpoint=%s",
            project_name,
            endpoint or "default",
        )
        return True

    except Exception as e:
        logger.warning("Failed to setup OpenInference tracing: %s", e)
        return False


def setup_console_tracing(
    project_name: str = "mask-agent",
) -> bool:
    """Set up console-based tracing for development.

    Prints trace information to stdout for debugging.

    Args:
        project_name: Name of the project.

    Returns:
        True if setup was successful, False otherwise.
    """
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except ImportError:
        logger.warning(
            "Console tracing requires opentelemetry packages. "
            "Install with: pip install mask-kernel[observability]"
        )
        return False

    try:
        # Create console exporter
        provider = TracerProvider()
        processor = SimpleSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=provider)

        logger.info("Console tracing enabled for %s", project_name)
        return True

    except Exception as e:
        logger.warning("Failed to setup console tracing: %s", e)
        return False


def disable_tracing() -> None:
    """Disable all tracing instrumentation.

    Use this to turn off tracing when no longer needed.
    """
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().uninstrument()
        logger.info("Tracing disabled")
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Failed to disable tracing: %s", e)
