"""MASK observability and tracing.

This module provides utilities for setting up observability
using Langfuse (recommended) or Phoenix/OpenInference (alternative).

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
"""

from mask.observability.setup import (
    # Langfuse (recommended)
    setup_langfuse_tracing,
    get_langfuse_client,
    get_langfuse_handler,
    shutdown_langfuse,
    # Phoenix (alternative)
    setup_openinference_tracing,
    setup_console_tracing,
    # Common
    disable_tracing,
)

__all__ = [
    # Langfuse (recommended)
    "setup_langfuse_tracing",
    "get_langfuse_client",
    "get_langfuse_handler",
    "shutdown_langfuse",
    # Phoenix (alternative)
    "setup_openinference_tracing",
    "setup_console_tracing",
    # Common
    "disable_tracing",
]
