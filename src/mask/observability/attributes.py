"""Multi-backend span attribute utilities.

Provides unified attribute setting for multiple observability backends:
- Phoenix (OpenInference)
- Langfuse
- OpenTelemetry GenAI semantic conventions

Usage:
    from mask.observability.attributes import set_span_io, set_span_session

    with tracer.start_as_current_span("my_span") as span:
        set_span_io(span, input_value="Hello", output_value="World")
        set_span_session(span, session_id="abc123", user_id="user1")
"""

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def set_span_io(
    span,
    input_value: Optional[str] = None,
    output_value: Optional[str] = None,
    input_mime_type: str = "text/plain",
    output_mime_type: str = "text/plain",
) -> None:
    """Set input/output attributes on span for multiple backends.

    Sets attributes in formats supported by:
    - Phoenix/OpenInference: input.value, output.value
    - Langfuse: langfuse.observation.input, langfuse.observation.output
    - OpenTelemetry GenAI: gen_ai.prompt, gen_ai.completion

    Args:
        span: OpenTelemetry Span object
        input_value: Input text/data
        output_value: Output text/data
        input_mime_type: MIME type for input (default: text/plain)
        output_mime_type: MIME type for output (default: text/plain)
    """
    if not span or not span.is_recording():
        return

    if input_value is not None:
        # OpenInference (Phoenix)
        span.set_attribute("input.value", input_value)
        span.set_attribute("input.mime_type", input_mime_type)

        # Langfuse
        span.set_attribute("langfuse.observation.input", input_value)

        # OpenTelemetry GenAI
        span.set_attribute("gen_ai.prompt", input_value)

    if output_value is not None:
        # OpenInference (Phoenix)
        span.set_attribute("output.value", output_value)
        span.set_attribute("output.mime_type", output_mime_type)

        # Langfuse
        span.set_attribute("langfuse.observation.output", output_value)

        # OpenTelemetry GenAI
        span.set_attribute("gen_ai.completion", output_value)


def set_span_session(
    span,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    trace_name: Optional[str] = None,
) -> None:
    """Set session/user attributes on span for multiple backends.

    Sets attributes in formats supported by:
    - Phoenix/OpenInference: session.id
    - Langfuse: langfuse.session.id, langfuse.user.id, langfuse.trace.name

    Args:
        span: OpenTelemetry Span object
        session_id: Session/conversation identifier
        user_id: User identifier
        trace_name: Name for the trace (Langfuse-specific)
    """
    if not span or not span.is_recording():
        return

    if session_id is not None:
        # OpenInference (Phoenix)
        span.set_attribute("session.id", session_id)

        # Langfuse
        span.set_attribute("langfuse.session.id", session_id)

    if user_id is not None:
        # OpenInference (Phoenix) - no direct equivalent
        span.set_attribute("user.id", user_id)

        # Langfuse
        span.set_attribute("langfuse.user.id", user_id)

    if trace_name is not None:
        # Langfuse-specific
        span.set_attribute("langfuse.trace.name", trace_name)


def set_span_model(
    span,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    cost: Optional[float] = None,
) -> None:
    """Set model/LLM attributes on span for multiple backends.

    Sets attributes in formats supported by:
    - OpenInference: llm.model_name, llm.token_count.*
    - Langfuse: langfuse.observation.model.name, langfuse.observation.usage_details
    - OpenTelemetry GenAI: gen_ai.request.model, gen_ai.usage.*

    Args:
        span: OpenTelemetry Span object
        model_name: Name of the LLM model
        provider: Model provider (e.g., "anthropic", "openai")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        total_tokens: Total tokens used
        cost: Cost in USD
    """
    if not span or not span.is_recording():
        return

    if model_name is not None:
        # OpenInference
        span.set_attribute("llm.model_name", model_name)

        # Langfuse
        span.set_attribute("langfuse.observation.model.name", model_name)

        # OpenTelemetry GenAI
        span.set_attribute("gen_ai.request.model", model_name)
        span.set_attribute("gen_ai.response.model", model_name)

    if provider is not None:
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("llm.provider", provider)

    # Token counts
    if input_tokens is not None:
        span.set_attribute("llm.token_count.prompt", input_tokens)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)

    if output_tokens is not None:
        span.set_attribute("llm.token_count.completion", output_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    if total_tokens is not None:
        span.set_attribute("llm.token_count.total", total_tokens)

    # Langfuse usage details (JSON format)
    if any([input_tokens, output_tokens, total_tokens]):
        usage_details = {}
        if input_tokens is not None:
            usage_details["input"] = input_tokens
        if output_tokens is not None:
            usage_details["output"] = output_tokens
        if total_tokens is not None:
            usage_details["total"] = total_tokens
        span.set_attribute(
            "langfuse.observation.usage_details",
            json.dumps(usage_details)
        )

    if cost is not None:
        span.set_attribute("gen_ai.usage.cost", cost)
        span.set_attribute(
            "langfuse.observation.cost_details",
            json.dumps({"total": cost})
        )


def set_span_metadata(
    span,
    agent_name: Optional[str] = None,
    server_name: Optional[str] = None,
    environment: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[list[str]] = None,
    **extra_metadata: Any,
) -> None:
    """Set metadata attributes on span for multiple backends.

    Args:
        span: OpenTelemetry Span object
        agent_name: Name of the agent
        server_name: Name of the A2A server
        environment: Deployment environment (e.g., "production", "staging")
        version: Application/agent version
        tags: List of tags for categorization
        **extra_metadata: Additional custom metadata
    """
    if not span or not span.is_recording():
        return

    if agent_name is not None:
        span.set_attribute("mask.agent.name", agent_name)

    if server_name is not None:
        span.set_attribute("mask.server.name", server_name)

    if environment is not None:
        span.set_attribute("deployment.environment", environment)
        span.set_attribute("langfuse.environment", environment)

    if version is not None:
        span.set_attribute("service.version", version)
        span.set_attribute("langfuse.version", version)

    if tags is not None:
        # Langfuse supports array of strings
        span.set_attribute("langfuse.trace.tags", tags)

    # Set any extra metadata with langfuse prefix
    for key, value in extra_metadata.items():
        if value is not None:
            span.set_attribute(f"langfuse.trace.metadata.{key}", str(value))
            span.set_attribute(f"mask.metadata.{key}", str(value))
