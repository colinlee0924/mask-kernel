"""Example: Orchestrator Agent with Sub-Agent Delegation.

This example demonstrates how to build an orchestrator agent that:
1. Delegates tasks to sub-agents via A2A protocol
2. Streams events from sub-agents to the frontend
3. Uses A2AStreamingMiddleware for real-time status updates

Architecture:
    ┌─────────────────┐
    │   Open WebUI    │◄──── SSE streaming
    └────────▲────────┘
             │
    ┌────────┴────────┐
    │   Orchestrator  │◄──── A2AStreamingMiddleware
    │   (this agent)  │       emits thinking/tool events
    └────────▲────────┘
             │ A2A sendSubscribe (SSE)
             ▼
    ┌─────────────────┐    ┌─────────────────┐
    │   JIRA Agent    │    │    FAQ Agent    │
    │  (port 10002)   │    │  (port 10003)   │
    └─────────────────┘    └─────────────────┘

Usage:
    # Start sub-agents first
    python -m jira_agent.main  # Port 10002
    python -m faq_agent.main   # Port 10003

    # Then start orchestrator
    python examples/orchestrator_agent.py  # Port 10001

Environment Variables:
    ANTHROPIC_API_KEY: Anthropic API key
    A2A_HOST: Host to bind (default: 0.0.0.0)
    A2A_PORT: Port to bind (default: 10001)
    JIRA_AGENT_URL: JIRA agent URL (default: http://localhost:10002)
    FAQ_AGENT_URL: FAQ agent URL (default: http://localhost:10003)
"""

import asyncio
import logging
import os
from typing import List

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from mask.a2a import (
    DelegationToolFactory,
    create_a2a_executor,
    create_delegation_tools,
)
from mask.middleware import A2AStreamingMiddleware, SkillMiddleware
from mask.observability import setup_openinference_tracing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
A2A_HOST = os.getenv("A2A_HOST", "0.0.0.0")
A2A_PORT = int(os.getenv("A2A_PORT", "10001"))
JIRA_AGENT_URL = os.getenv("JIRA_AGENT_URL", "http://localhost:10002")
FAQ_AGENT_URL = os.getenv("FAQ_AGENT_URL", "http://localhost:10003")


SYSTEM_PROMPT = """You are an orchestrator agent that coordinates multiple specialized agents.

Available sub-agents:
- jira-agent: Handles JIRA-related tasks (search, create, update tickets)
- faq-agent: Answers frequently asked questions

When a user request requires specialized capabilities:
1. Analyze the request to determine which agent(s) to use
2. Delegate to the appropriate agent using the delegation tools
3. Synthesize the results into a coherent response

Always explain what you're doing before delegating, and summarize the results after.
"""


async def create_orchestrator_agent(
    delegation_tools: List[BaseTool],
    streaming_middleware: A2AStreamingMiddleware,
):
    """Create the orchestrator agent with delegation tools.

    Args:
        delegation_tools: Tools for delegating to sub-agents.
        streaming_middleware: Middleware for event streaming.

    Returns:
        Compiled agent graph.
    """
    try:
        # Try LangChain v1.x create_agent
        from langchain.agents import create_agent

        model = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

        agent = create_agent(
            model=model,
            tools=delegation_tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[streaming_middleware],
        )

        logger.info("Created orchestrator agent with LangChain create_agent")
        return agent

    except ImportError:
        # Fallback to LangGraph create_react_agent
        from langgraph.prebuilt import create_react_agent

        model = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

        agent = create_react_agent(
            model=model,
            tools=delegation_tools,
            state_modifier=SYSTEM_PROMPT,
        )

        logger.info("Created orchestrator agent with LangGraph create_react_agent")
        return agent


async def setup_agent():
    """Set up the orchestrator agent with delegation tools."""
    # Set up observability
    setup_openinference_tracing(project_name="orchestrator-agent")

    # Create streaming middleware
    streaming_middleware = A2AStreamingMiddleware(
        agent_name="orchestrator",
        emit_thinking=True,
    )

    # Create delegation tool factory
    factory = DelegationToolFactory(track_delegation_history=True)

    # Register sub-agents (comment out unavailable ones)
    try:
        await factory.register_agent(
            JIRA_AGENT_URL,
            name="jira-agent",
            description="Handles JIRA-related tasks like searching, creating, and updating tickets",
        )
        logger.info("Registered jira-agent at %s", JIRA_AGENT_URL)
    except Exception as e:
        logger.warning("Failed to register jira-agent: %s", e)

    try:
        await factory.register_agent(
            FAQ_AGENT_URL,
            name="faq-agent",
            description="Answers frequently asked questions from the knowledge base",
        )
        logger.info("Registered faq-agent at %s", FAQ_AGENT_URL)
    except Exception as e:
        logger.warning("Failed to register faq-agent: %s", e)

    # Get delegation tools
    delegation_tools = factory.get_tools()
    logger.info("Created %d delegation tools: %s", len(delegation_tools), [t.name for t in delegation_tools])

    if not delegation_tools:
        logger.warning("No sub-agents registered. Add local tools for testing.")
        # Add a simple local tool for testing
        from langchain_core.tools import tool

        @tool
        def echo(message: str) -> str:
            """Echo the message back. Use this for testing."""
            return f"Echo: {message}"

        delegation_tools = [echo]

    # Create agent
    agent = await create_orchestrator_agent(delegation_tools, streaming_middleware)

    return agent, streaming_middleware, factory


def create_agent_card() -> AgentCard:
    """Create the A2A AgentCard for this orchestrator."""
    return AgentCard(
        name="orchestrator-agent",
        description="Orchestrator agent that coordinates multiple specialized agents",
        url=f"http://{A2A_HOST}:{A2A_PORT}",
        version="1.0.0",
        skills=[
            AgentSkill(
                id="orchestration",
                name="Task Orchestration",
                description="Coordinates multiple specialized agents to complete complex tasks",
                tags=["orchestrator", "multi-agent"],
            ),
        ],
        capabilities=AgentCapabilities(streaming=True),
    )


async def main():
    """Main entry point."""
    logger.info("Starting Orchestrator Agent...")
    logger.info("Sub-agents: JIRA=%s, FAQ=%s", JIRA_AGENT_URL, FAQ_AGENT_URL)

    # Set up agent
    agent, streaming_middleware, factory = await setup_agent()

    # Create A2A executor with streaming middleware
    executor = create_a2a_executor(
        agent,
        server_name="orchestrator-agent",
        streaming_middleware=streaming_middleware,
    )

    # Create task store and request handler
    task_store = InMemoryTaskStore()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    # Create A2A application
    agent_card = create_agent_card()
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    logger.info("Orchestrator Agent ready at http://%s:%s", A2A_HOST, A2A_PORT)

    # Run server
    config = uvicorn.Config(
        app=app.build(),
        host=A2A_HOST,
        port=A2A_PORT,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
