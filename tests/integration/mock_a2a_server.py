"""Mock A2A server for integration testing.

This module provides a mock A2A server that simulates:
- Agent card endpoint (/.well-known/agent.json)
- sendSubscribe endpoint with SSE streaming

Usage:
    from tests.integration.mock_a2a_server import app

    # Use with httpx ASGITransport for testing
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test code here
"""

import asyncio
import json
from typing import AsyncGenerator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


# Agent card for discovery
MOCK_AGENT_CARD = {
    "name": "mock-agent",
    "description": "A mock A2A agent for testing",
    "url": "http://localhost:10099",
    "version": "1.0.0",
    "capabilities": {"streaming": True},
    "skills": [
        {
            "id": "general",
            "name": "General",
            "description": "General assistance",
            "tags": ["general"],
        }
    ],
}


async def agent_card_endpoint(request: Request) -> JSONResponse:
    """Return the agent card for discovery."""
    return JSONResponse(MOCK_AGENT_CARD)


async def generate_sse_events() -> AsyncGenerator[str, None]:
    """Generate mock SSE events for testing.

    Simulates a typical agent execution flow:
    1. Status: working
    2. Tool call artifact
    3. Tool result artifact
    4. Response artifact
    5. Status: completed
    """
    events = [
        # Initial status
        {
            "jsonrpc": "2.0",
            "result": {
                "state": "working",
                "message": {
                    "parts": [
                        {"text": "Starting processing..."},
                        {"data": {"event_type": "agent_start"}},
                    ]
                },
            },
        },
        # Thinking status
        {
            "jsonrpc": "2.0",
            "result": {
                "state": "working",
                "message": {
                    "parts": [
                        {"text": "Analyzing request..."},
                        {"data": {"event_type": "llm_thinking"}},
                    ]
                },
            },
        },
        # Tool start status
        {
            "jsonrpc": "2.0",
            "result": {
                "state": "working",
                "message": {
                    "parts": [
                        {"text": "Running tool: mock_search"},
                        {
                            "data": {
                                "event_type": "tool_start",
                                "tool_name": "mock_search",
                                "input": {"query": "test query"},
                            }
                        },
                    ]
                },
            },
        },
        # Tool call artifact
        {
            "jsonrpc": "2.0",
            "result": {
                "artifact": {
                    "artifact_id": "artifact-1",
                    "name": "tool_call",
                    "parts": [
                        {
                            "text": json.dumps(
                                {"tool": "mock_search", "input": {"query": "test query"}}
                            )
                        }
                    ],
                }
            },
        },
        # Tool end status
        {
            "jsonrpc": "2.0",
            "result": {
                "state": "working",
                "message": {
                    "parts": [
                        {"text": "Tool completed: mock_search"},
                        {
                            "data": {
                                "event_type": "tool_end",
                                "tool_name": "mock_search",
                                "output": "Found 3 results",
                                "duration_ms": 150,
                            }
                        },
                    ]
                },
            },
        },
        # Tool result artifact
        {
            "jsonrpc": "2.0",
            "result": {
                "artifact": {
                    "artifact_id": "artifact-2",
                    "name": "tool_result",
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "tool": "mock_search",
                                    "output": "Found 3 results",
                                    "duration_ms": 150,
                                }
                            )
                        }
                    ],
                }
            },
        },
        # Response artifact (streaming text)
        {
            "jsonrpc": "2.0",
            "result": {
                "artifact": {
                    "artifact_id": "artifact-3",
                    "name": "response",
                    "parts": [{"text": "Based on the search results, "}],
                }
            },
        },
        {
            "jsonrpc": "2.0",
            "result": {
                "artifact": {
                    "artifact_id": "artifact-3",
                    "name": "response",
                    "parts": [{"text": "here is my analysis."}],
                }
            },
        },
        # Completion status
        {
            "jsonrpc": "2.0",
            "result": {
                "state": "completed",
                "message": {
                    "parts": [
                        {"text": "Task completed"},
                        {"data": {"event_type": "agent_end"}},
                    ]
                },
            },
        },
    ]

    for event in events:
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0.01)  # Small delay between events


async def handle_json_rpc(request: Request):
    """Handle JSON-RPC requests."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}},
            status_code=400,
        )

    method = body.get("method", "")
    request_id = body.get("id", 1)

    if method == "tasks/sendSubscribe":
        # Check if client wants SSE
        accept = request.headers.get("accept", "")
        if "text/event-stream" in accept:
            return StreamingResponse(
                generate_sse_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming response
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "state": "completed",
                        "message": {
                            "parts": [{"text": "Non-streaming response"}]
                        },
                    },
                }
            )

    elif method == "tasks/send":
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "state": "completed",
                    "message": {"parts": [{"text": "Response from send"}]},
                },
            }
        )

    else:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            },
            status_code=400,
        )


# Create the Starlette app
app = Starlette(
    debug=True,
    routes=[
        Route("/.well-known/agent.json", agent_card_endpoint, methods=["GET"]),
        Route("/", handle_json_rpc, methods=["POST"]),
    ],
)


# Error simulation variants
async def generate_error_sse_events() -> AsyncGenerator[str, None]:
    """Generate SSE events that simulate an error."""
    events = [
        {
            "jsonrpc": "2.0",
            "result": {
                "state": "working",
                "message": {"parts": [{"text": "Starting..."}]},
            },
        },
        {
            "jsonrpc": "2.0",
            "result": {
                "state": "failed",
                "message": {"parts": [{"text": "An error occurred"}]},
            },
        },
    ]

    for event in events:
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0.01)


def create_mock_app_with_error():
    """Create a mock app that simulates errors."""

    async def error_handler(request: Request):
        body = await request.json()
        method = body.get("method", "")

        if method == "tasks/sendSubscribe":
            return StreamingResponse(
                generate_error_sse_events(),
                media_type="text/event-stream",
            )

        return JSONResponse({"error": "Not implemented"}, status_code=500)

    return Starlette(
        routes=[
            Route("/.well-known/agent.json", agent_card_endpoint, methods=["GET"]),
            Route("/", error_handler, methods=["POST"]),
        ]
    )
