"""OpenAI-Compatible API Wrapper for A2A Agent.

This module provides an OpenAI-compatible API endpoint that wraps
the A2A agent via HTTP calls, enabling integration with Open WebUI
or any OpenAI-compatible client.

Architecture:
    Client (Open WebUI, etc.) → OpenAI Wrapper → A2A Agent

Supported endpoints:
- GET /v1/models - List available models
- POST /v1/chat/completions - Chat completion with streaming support

Usage:
    from mask.a2a import create_openai_compat_app

    app = create_openai_compat_app(
        a2a_base_url="http://localhost:10001",
        model_name="my-agent",
    )

    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)

Open WebUI Configuration:
    1. Settings → Connections → OpenAI API
    2. Add new connection:
       - URL: http://localhost:11434/v1
       - API Key: sk-dummy (any value)
    3. Select your model in the chat
"""

import json
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, List, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from fastapi import FastAPI


class Message(BaseModel):
    """OpenAI-style message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-style chat completion request."""

    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None


class ChatCompletionChoice(BaseModel):
    """OpenAI-style choice."""

    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """OpenAI-style chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: dict


class ModelInfo(BaseModel):
    """OpenAI-style model info."""

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    """OpenAI-style models list response."""

    object: str = "list"
    data: List[ModelInfo]


def create_openai_compat_app(
    a2a_base_url: str = "http://localhost:10001",
    model_name: str = "mask-agent",
    owned_by: str = "mask-kernel",
) -> "FastAPI":
    """Create FastAPI app with OpenAI-compatible endpoints wrapping A2A.

    This creates an OpenAI-compatible API server that forwards requests
    to an A2A agent, enabling integration with clients like Open WebUI.

    Args:
        a2a_base_url: Base URL of the A2A agent server.
        model_name: Name to use for the model in API responses.
        owned_by: Organization name for model info.

    Returns:
        FastAPI application with /v1/models and /v1/chat/completions endpoints.

    Example:
        >>> from mask.a2a import create_openai_compat_app
        >>> app = create_openai_compat_app(
        ...     a2a_base_url="http://localhost:10001",
        ...     model_name="my-awesome-agent",
        ... )
        >>> # Run with: uvicorn.run(app, host="0.0.0.0", port=11434)
    """
    # Import here to make fastapi optional
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from starlette.middleware.cors import CORSMiddleware

    app = FastAPI(
        title=f"{model_name} OpenAI-Compatible API",
        description="OpenAI-compatible API wrapper for A2A agent",
        version="1.0.0",
    )

    # Add CORS middleware for Open WebUI
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models")
    async def list_models() -> ModelsResponse:
        """List available models (OpenAI-compatible)."""
        return ModelsResponse(
            data=[
                ModelInfo(
                    id=model_name,
                    created=int(time.time()),
                    owned_by=owned_by,
                )
            ]
        )

    @app.get("/models")
    async def list_models_alt() -> ModelsResponse:
        """List available models (alternative path)."""
        return await list_models()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Chat completion endpoint (OpenAI-compatible)."""
        # Extract the last user message
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # Use a consistent context_id for multi-turn conversations
        # Hash the conversation history to maintain context
        context_id = (
            f"openai-compat-{hash(tuple(m.content for m in request.messages)) % 10000000:07d}"
        )

        if request.stream:
            return StreamingResponse(
                _stream_response_via_a2a(
                    a2a_base_url,
                    user_message,
                    context_id,
                    completion_id,
                    created,
                    model_name,
                ),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            response_text = await _invoke_a2a_agent(
                a2a_base_url, user_message, context_id
            )
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(role="assistant", content=response_text),
                        finish_reason="stop",
                    )
                ],
                usage={
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(user_message.split())
                    + len(response_text.split()),
                },
            )

    @app.post("/chat/completions")
    async def chat_completions_alt(request: ChatCompletionRequest):
        """Chat completion endpoint (alternative path)."""
        return await chat_completions(request)

    return app


async def _invoke_a2a_agent(
    a2a_base_url: str,
    message: str,
    context_id: str,
) -> str:
    """Invoke A2A agent and get response.

    Args:
        a2a_base_url: Base URL of the A2A agent server.
        message: User message to send.
        context_id: Context ID for multi-turn conversation.

    Returns:
        Response text from the agent.
    """
    import httpx

    message_id = f"msg-{uuid.uuid4().hex[:8]}"

    a2a_request = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": "1",
        "params": {
            "message": {
                "messageId": message_id,
                "contextId": context_id,
                "role": "user",
                "parts": [{"text": message}],
            }
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            a2a_base_url,
            json=a2a_request,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        result = response.json()

        # Extract text from A2A response
        if "result" in result and "parts" in result["result"]:
            parts = result["result"]["parts"]
            text_parts = []
            for part in parts:
                if part.get("kind") == "text" and "text" in part:
                    text_parts.append(part["text"])
            return "".join(text_parts)

        return ""


async def _stream_response_via_a2a(
    a2a_base_url: str,
    message: str,
    context_id: str,
    completion_id: str,
    created: int,
    model: str,
) -> AsyncGenerator[str, None]:
    """Stream response via A2A in OpenAI SSE format.

    Note: A2A streaming is handled server-side. This function calls A2A
    and then streams the response back in OpenAI SSE format.

    For true end-to-end streaming, A2A SSE endpoint would be needed.
    """
    try:
        # Call A2A agent
        response_text = await _invoke_a2a_agent(a2a_base_url, message, context_id)

        # Stream the response in chunks for real-time effect
        # This simulates streaming even though A2A returns full response
        chunk_size = 20  # Characters per chunk

        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i : i + chunk_size]
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

        # Send final chunk with finish_reason
        final_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"


def run_openai_compat_server(
    a2a_base_url: str = "http://localhost:10001",
    model_name: str = "mask-agent",
    host: str = "0.0.0.0",
    port: int = 11434,
) -> None:
    """Run OpenAI-compatible wrapper server.

    Convenience function to start the server with uvicorn.

    Args:
        a2a_base_url: Base URL of the A2A agent server.
        model_name: Name to use for the model in API responses.
        host: Host to bind to.
        port: Port to listen on.

    Example:
        >>> from mask.a2a import run_openai_compat_server
        >>> run_openai_compat_server(
        ...     a2a_base_url="http://localhost:10001",
        ...     model_name="my-agent",
        ...     port=11434,
        ... )
    """
    import uvicorn

    app = create_openai_compat_app(
        a2a_base_url=a2a_base_url,
        model_name=model_name,
    )

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         OpenAI-Compatible Wrapper for A2A Agent                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Wrapper:  http://{host}:{port:<5}                                 ║
║  A2A:      {a2a_base_url:<20}                           ║
║                                                                  ║
║  Open WebUI Configuration:                                       ║
║  ─────────────────────────                                       ║
║  1. Settings → Connections → OpenAI API                          ║
║  2. Click "+" to add connection:                                 ║
║     • URL: http://localhost:{port}/v1                             ║
║     • API Key: sk-dummy (any value)                              ║
║  3. Select "{model_name}" in Models                     ║
║                                                                  ║
║  Endpoints:                                                      ║
║  • GET  /v1/models          - List available models              ║
║  • POST /v1/chat/completions - Chat (supports streaming)         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host=host, port=port, log_level="info")
