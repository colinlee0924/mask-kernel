"""MASK observability and tracing.

This module provides utilities for setting up observability
using Langfuse (recommended) or Phoenix/OpenInference (alternative).

Both backends are supported through a unified attribute abstraction layer
that sets compatible attributes for Phoenix, Langfuse, and OpenTelemetry GenAI.

Langfuse (recommended):
    pip install mask-kernel[observability]

Phoenix (alternative):
    pip install mask-kernel[phoenix]

Example (Langfuse):
    from mask.observability import setup_langfuse_tracing, get_langfuse_handler

    # Setup tracing
    langfuse = setup_langfuse_tracing()

    # Get handler for LangChain/LangGraph
    handler = get_langfuse_handler()
    response = graph.invoke({"messages": [...]}, config={"callbacks": [handler]})

Example (Phoenix):
    from mask.observability import setup_openinference_tracing

    setup_openinference_tracing("my-agent")

Example (Custom Span Attributes):
    from mask.observability import set_span_io, set_span_session, set_span_model

    with tracer.start_as_current_span("my_span") as span:
        set_span_io(span, input_value="Hello", output_value="World")
        set_span_session(span, session_id="abc123", user_id="user1")
        set_span_model(span, model_name="claude-3", input_tokens=100)
"""

from mask.observability.setup import (
    # Langfuse
    setup_langfuse_tracing,
    setup_langfuse_otel_tracing,
    get_langfuse_client,
    get_langfuse_handler,
    shutdown_langfuse,
    # Phoenix
    setup_openinference_tracing,
    setup_console_tracing,
    # Dual (Phoenix + Langfuse)
    setup_dual_tracing,
    # Common
    disable_tracing,
)

from mask.observability.attributes import (
    # Multi-backend attribute utilities
    set_span_io,
    set_span_session,
    set_span_model,
    set_span_metadata,
)

__all__ = [
    # Langfuse
    "setup_langfuse_tracing",
    "setup_langfuse_otel_tracing",
    "get_langfuse_client",
    "get_langfuse_handler",
    "shutdown_langfuse",
    # Phoenix
    "setup_openinference_tracing",
    "setup_console_tracing",
    # Dual (Phoenix + Langfuse)
    "setup_dual_tracing",
    # Common
    "disable_tracing",
    # Multi-backend attribute utilities
    "set_span_io",
    "set_span_session",
    "set_span_model",
    "set_span_metadata",
]
