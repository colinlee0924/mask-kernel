"""MASK observability and tracing.

This module provides utilities for setting up observability
using OpenInference and Arize Phoenix.

Requires: pip install mask-kernel[observability]
"""

from mask.observability.setup import (
    disable_tracing,
    setup_console_tracing,
    setup_openinference_tracing,
)

__all__ = [
    "setup_openinference_tracing",
    "setup_console_tracing",
    "disable_tracing",
]
