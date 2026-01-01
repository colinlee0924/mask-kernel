"""Observability setup for MASK Kernel.

This module provides utilities for setting up observability and tracing
using Langfuse (recommended) or Phoenix/OpenInference (alternative).

Langfuse (recommended):
    pip install mask-kernel[observability]

Phoenix (alternative):
    pip install mask-kernel[phoenix]
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

# Singleton for Langfuse client
_langfuse_client: Optional["Langfuse"] = None


# =============================================================================
# Langfuse (Recommended)
# =============================================================================


def setup_langfuse_tracing(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    base_url: Optional[str] = None,
    debug: bool = False,
) -> Optional["Langfuse"]:
    """Set up Langfuse tracing for observability.

    Initializes the Langfuse client for tracing LangChain/LangGraph operations.
    Uses singleton pattern - subsequent calls return the same client.

    Args:
        public_key: Langfuse public key. If not provided, uses
            LANGFUSE_PUBLIC_KEY environment variable.
        secret_key: Langfuse secret key. If not provided, uses
            LANGFUSE_SECRET_KEY environment variable.
        base_url: Langfuse server URL. If not provided, uses
            LANGFUSE_BASE_URL environment variable or defaults to cloud.
        debug: Enable debug logging.

    Returns:
        Langfuse client instance if successful, None otherwise.

    Example:
        from mask.observability import setup_langfuse_tracing, get_langfuse_handler

        # Setup tracing (reads from env vars)
        langfuse = setup_langfuse_tracing()

        # Get handler for LangChain/LangGraph
        handler = get_langfuse_handler()
        response = graph.invoke({"messages": [...]}, config={"callbacks": [handler]})
    """
    global _langfuse_client

    # Return existing client if already initialized
    if _langfuse_client is not None:
        return _langfuse_client

    # Try to import langfuse
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning(
            "Langfuse is not installed. "
            "Install with: pip install mask-kernel[observability]"
        )
        return None

    # Get credentials from args or environment
    public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    base_url = base_url or os.environ.get("LANGFUSE_BASE_URL")

    # Check if credentials are provided
    if not public_key or not secret_key:
        logger.warning(
            "Langfuse credentials not configured. "
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
        )
        return None

    try:
        # Initialize Langfuse client
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=base_url,  # Langfuse uses 'host' parameter, not 'base_url'
            debug=debug,
        )

        logger.info(
            "Langfuse tracing enabled: base_url=%s",
            base_url or "https://cloud.langfuse.com",
        )
        return _langfuse_client

    except Exception as e:
        logger.warning("Failed to setup Langfuse tracing: %s", e)
        return None


def get_langfuse_client() -> Optional["Langfuse"]:
    """Get the initialized Langfuse client.

    Returns the singleton Langfuse client if it has been initialized via
    setup_langfuse_tracing(), otherwise returns None.

    Returns:
        Langfuse client instance or None if not initialized.

    Example:
        from mask.observability import get_langfuse_client

        langfuse = get_langfuse_client()
        if langfuse:
            trace_url = langfuse.get_trace_url()
            trace_id = langfuse.get_current_trace_id()
    """
    return _langfuse_client


def get_langfuse_handler() -> Optional["CallbackHandler"]:
    """Get the Langfuse CallbackHandler for LangChain/LangGraph.

    Creates a new CallbackHandler instance for use with LangChain/LangGraph.
    Returns None if Langfuse is not installed or not configured.

    Returns:
        CallbackHandler instance or None if not available.

    Example:
        from mask.observability import get_langfuse_handler

        handler = get_langfuse_handler()
        callbacks = [handler] if handler else []
        response = graph.invoke({"messages": [...]}, config={"callbacks": callbacks})
    """
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.debug("Langfuse not installed, returning None handler")
        return None

    # Check if credentials are available
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        logger.debug("Langfuse credentials not configured, returning None handler")
        return None

    try:
        return CallbackHandler()
    except Exception as e:
        logger.warning("Failed to create Langfuse handler: %s", e)
        return None


def shutdown_langfuse() -> None:
    """Shutdown Langfuse client and flush pending data.

    Call this before process exit to ensure all traces are sent.
    """
    global _langfuse_client

    if _langfuse_client is not None:
        try:
            _langfuse_client.shutdown()
            logger.info("Langfuse client shutdown complete")
        except Exception as e:
            logger.warning("Error during Langfuse shutdown: %s", e)
        finally:
            _langfuse_client = None


# =============================================================================
# Phoenix/OpenInference (Alternative)
# =============================================================================


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
            "Phoenix/OpenInference is not installed. "
            "Install with: pip install mask-kernel[phoenix]"
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
            "Install with: pip install mask-kernel[phoenix]"
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
    # Shutdown Langfuse
    shutdown_langfuse()

    # Disable OpenInference
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().uninstrument()
        logger.info("OpenInference tracing disabled")
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Failed to disable OpenInference tracing: %s", e)
