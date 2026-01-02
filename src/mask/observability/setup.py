"""Observability setup for MASK Kernel.

This module provides utilities for setting up observability and tracing
using Phoenix/OpenInference (recommended) or Langfuse (optional).

Phoenix (recommended):
    pip install mask-kernel[phoenix]

Langfuse (optional):
    pip install mask-kernel[observability]
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

# =============================================================================
# LangChain Observability Patch
# =============================================================================

# CRITICAL: Monkey patch to enable tracing in LangChain agents
# LangChain's create_agent uses RunnableCallable(trace=False) which prevents
# callback handlers (including Phoenix OpenInferenceTracer and Langfuse CallbackHandler)
# from capturing detailed execution traces. This patch forces trace=True globally.
try:
    from langgraph._internal._runnable import RunnableCallable

    _original_runnable_callable_init = RunnableCallable.__init__

    def _patched_runnable_callable_init(self, *args, trace=True, **kwargs):
        """Force trace=True for all RunnableCallable instances to enable observability."""
        # Always use trace=True, ignoring the original parameter
        _original_runnable_callable_init(self, *args, trace=True, **kwargs)

    # Apply the monkey patch
    RunnableCallable.__init__ = _patched_runnable_callable_init
    logger.debug("Applied monkey patch to RunnableCallable to enable tracing")
except ImportError:
    # LangGraph not installed, skip monkey patch
    logger.debug("LangGraph not installed, skipping RunnableCallable monkey patch")
except Exception as e:
    logger.warning(f"Failed to apply RunnableCallable monkey patch: {e}")

# Singleton for Langfuse client
_langfuse_client: Optional["Langfuse"] = None


# =============================================================================
# Langfuse (Optional)
# =============================================================================


def setup_langfuse_tracing(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    base_url: Optional[str] = None,
    debug: bool = False,
    blocked_scopes: Optional[list] = None,
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
        blocked_scopes: List of instrumentation scope names to filter out.
            Default blocks A2A SDK traces to reduce noise.

    Returns:
        Langfuse client instance if successful, None otherwise.

    Example:
        from mask.observability import setup_langfuse_tracing, get_langfuse_handler

        # Setup tracing (reads from env vars, blocks A2A noise by default)
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

    # Default blocked scopes: A2A SDK infrastructure traces
    if blocked_scopes is None:
        blocked_scopes = [
            "a2a-python-sdk",  # A2A SDK internal traces
        ]

    try:
        # Initialize Langfuse client with scope filtering
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=base_url,  # Langfuse uses 'host' parameter, not 'base_url'
            debug=debug,
            blocked_instrumentation_scopes=blocked_scopes,
        )

        logger.info(
            "Langfuse tracing enabled: base_url=%s, blocked_scopes=%s",
            base_url or "https://cloud.langfuse.com",
            blocked_scopes,
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


def get_langfuse_handler(
    trace_name: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    tags: Optional[list] = None,
    metadata: Optional[dict] = None,
) -> Optional["CallbackHandler"]:
    """Get the Langfuse CallbackHandler for LangChain/LangGraph.

    Creates a new CallbackHandler instance for use with LangChain/LangGraph.
    Returns None if Langfuse is not installed or not configured.

    Args:
        trace_name: Name for the trace (e.g., agent name). This appears as the
            top-level span name in Langfuse. Highly recommended for clarity.
        session_id: Session ID for grouping traces (like a thread ID).
        user_id: User ID for the trace.
        tags: List of tags for filtering traces.
        metadata: Additional metadata for the trace.

    Returns:
        CallbackHandler instance or None if not available.

    Example:
        from mask.observability import get_langfuse_handler

        handler = get_langfuse_handler(
            trace_name="my-agent",
            session_id="session-123",
        )
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
        # Create basic handler - trace metadata is set via config in LangGraph
        # In Langfuse SDK v3, the CallbackHandler auto-detects credentials
        # and trace structure comes from LangChain/LangGraph spans
        handler = CallbackHandler()

        # Store metadata for later use if needed
        if trace_name or session_id or user_id or tags or metadata:
            handler._mask_metadata = {
                "trace_name": trace_name,
                "session_id": session_id,
                "user_id": user_id,
                "tags": tags or [],
                "metadata": metadata or {},
            }

        return handler
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
# Phoenix/OpenInference (Recommended)
# =============================================================================


class FilteringSpanProcessor:
    """過濾特定 instrumentation scopes 的 spans，減少觀測雜訊。

    用於過濾 A2A SDK 等基礎設施層級的 traces，讓開發者專注於業務邏輯。

    Example:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
        base_processor = BatchSpanProcessor(exporter)
        filtering_processor = FilteringSpanProcessor(
            delegate_processor=base_processor,
            excluded_scopes=['a2a-python-sdk', 'opentelemetry']
        )
    """

    def __init__(self, delegate_processor, excluded_scopes=None):
        """初始化過濾 processor。

        Args:
            delegate_processor: 下游 SpanProcessor（如 BatchSpanProcessor）
            excluded_scopes: 要過濾的 instrumentation scope 名稱列表
        """
        self.delegate_processor = delegate_processor
        self.excluded_scopes = excluded_scopes or ['a2a-python-sdk']

    def on_start(self, span, parent_context=None):
        """Span 開始時調用（不需要過濾）"""
        pass

    def on_end(self, span):
        """Span 結束時調用，根據 scope 決定是否傳遞給下游。

        Args:
            span: ReadableSpan 對象，包含 instrumentation_scope 屬性
        """
        # 檢查 instrumentation_scope
        if span.instrumentation_scope:
            scope_name = span.instrumentation_scope.name
            if scope_name in self.excluded_scopes:
                logger.debug(
                    "Filtering span from scope: %s (name: %s)",
                    scope_name,
                    span.name
                )
                return  # 跳過這個 span，不傳給下游 processor

        # 將 span 傳給下游 processor
        self.delegate_processor.on_end(span)

    def shutdown(self):
        """關閉 processor"""
        return self.delegate_processor.shutdown()

    def force_flush(self, timeout_millis=30000):
        """強制刷新所有待處理的 spans"""
        return self.delegate_processor.force_flush(timeout_millis)


def setup_openinference_tracing(
    project_name: str = "mask-agent",
    endpoint: Optional[str] = None,
    batch: bool = True,
    excluded_scopes: Optional[list] = None,
) -> bool:
    """Set up OpenInference tracing with Arize Phoenix.

    Uses manual OpenTelemetry setup (same as basic-observability-examples).

    Args:
        project_name: Name of the project for trace grouping in Phoenix UI.
        endpoint: Phoenix endpoint URL. If not provided, uses
            PHOENIX_COLLECTOR_ENDPOINT env var or defaults to local.
        batch: Whether to batch trace exports (recommended for production).
        excluded_scopes: Currently ignored (simplified for stability).

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
        from opentelemetry import trace as trace_api
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "Phoenix/OpenInference is not installed. "
            "Install with: pip install mask-kernel[phoenix]"
        )
        return False

    # Determine endpoint
    if endpoint is None:
        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

    try:
        # Try to use Phoenix PROJECT_NAME constant for proper project name
        try:
            from phoenix.otel import PROJECT_NAME
            resource = Resource({PROJECT_NAME: project_name})
            logger.debug("Using Phoenix PROJECT_NAME attribute")
        except ImportError:
            # Fallback to service.name (will go to 'default' project)
            resource = Resource({"service.name": project_name})
            logger.debug("Phoenix otel not available, using service.name")

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)

        # Create exporter
        exporter = OTLPSpanExporter(
            endpoint=f"{endpoint}/v1/traces",
            headers={},
        )

        # Add batch processor
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as global tracer provider
        trace_api.set_tracer_provider(tracer_provider)

        # Instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        logger.info(
            "Phoenix tracing enabled: project=%s, endpoint=%s",
            project_name,
            endpoint,
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
