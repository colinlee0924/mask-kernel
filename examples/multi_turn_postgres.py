"""Multi-turn conversation with PostgreSQL persistence.

This example demonstrates how to use PostgreSQL checkpointer for
persistent multi-turn conversations with MASK agents.

Features demonstrated:
- PostgreSQL-backed conversation persistence
- Session resumption across API calls
- Skill state preservation
- Thread-based session management

Requirements:
    pip install mask-kernel[postgresql]
    pip install langgraph-checkpoint-postgres psycopg[binary,pool]

    # Start PostgreSQL (using Docker)
    docker run -d --name mask-postgres \
        -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_DB=mask_db \
        -p 5432:5432 \
        postgres:15

Usage:
    # Set environment variables
    export ANTHROPIC_API_KEY=your-key
    export POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/mask_db

    # Run example
    python examples/multi_turn_postgres.py
"""

import asyncio
import os
from pathlib import Path

from langchain_core.messages import HumanMessage


async def main():
    """Demonstrate multi-turn conversation with PostgreSQL persistence."""
    # Import MASK components
    from mask.agent import create_mask_agent
    from mask.checkpointer import create_postgres_checkpointer, create_thread_config
    from mask.core import SkillRegistry
    from mask.models import LLMFactory, ModelTier

    # Configuration
    postgres_url = os.environ.get(
        "POSTGRES_URL",
        "postgresql://postgres:postgres@localhost:5432/mask_db"
    )

    print("=" * 60)
    print("MASK Multi-Turn Conversation with PostgreSQL")
    print("=" * 60)
    print()

    # Create checkpointer
    print("Connecting to PostgreSQL...")
    checkpointer = await create_postgres_checkpointer(postgres_url)
    print("Connected!")
    print()

    try:
        # Create skill registry (with example skills if available)
        registry = SkillRegistry()
        skills_dir = Path(__file__).parent / "skills"
        if skills_dir.exists():
            count = registry.discover_from_directory(skills_dir)
            print(f"Discovered {count} skills")

        # Create LLM
        factory = LLMFactory()
        model = factory.get_model(tier=ModelTier.THINKING)

        # Create agent
        agent = create_mask_agent(
            model=model,
            skill_registry=registry,
            system_prompt="You are a helpful assistant with memory. Remember what the user tells you.",
        )

        # Build graph with checkpointer
        # Note: This requires the agent to expose its graph builder
        # For now, we'll demonstrate the checkpointer pattern

        # Simulate a thread ID (in production, use user_id + session_id)
        thread_id = "demo-session-001"
        config = create_thread_config(thread_id)

        print(f"Thread ID: {thread_id}")
        print()

        # --- First conversation turn ---
        print("-" * 40)
        print("Turn 1: User introduces themselves")
        print("-" * 40)

        # In a real implementation with checkpointer:
        # response = await graph.ainvoke(
        #     {"messages": [HumanMessage(content="My name is Alice and I work at Acme Corp")]},
        #     config=config,
        # )

        # For demo, use direct agent invoke
        response1 = await agent.invoke("My name is Alice and I work at Acme Corp")
        print(f"User: My name is Alice and I work at Acme Corp")
        print(f"Agent: {response1}")
        print()

        # --- Second conversation turn ---
        print("-" * 40)
        print("Turn 2: User asks about skills")
        print("-" * 40)

        response2 = await agent.invoke("What skills do you have available?")
        print(f"User: What skills do you have available?")
        print(f"Agent: {response2}")
        print()

        # --- Third conversation turn (memory test) ---
        print("-" * 40)
        print("Turn 3: Test memory recall")
        print("-" * 40)

        response3 = await agent.invoke("What's my name and where do I work?")
        print(f"User: What's my name and where do I work?")
        print(f"Agent: {response3}")
        print()

        print("=" * 60)
        print("Demo complete!")
        print()
        print("In production, use the checkpointer like this:")
        print()
        print("  from langgraph.graph import StateGraph")
        print("  from mask.checkpointer import create_postgres_checkpointer")
        print()
        print("  checkpointer = await create_postgres_checkpointer(postgres_url)")
        print("  graph = graph_builder.compile(checkpointer=checkpointer.checkpointer)")
        print()
        print("  # Each invoke with same thread_id continues the conversation")
        print("  config = {'configurable': {'thread_id': 'user-123'}}")
        print("  response = await graph.ainvoke({'messages': [...]}, config=config)")
        print("=" * 60)

    finally:
        # Clean up
        await checkpointer.close()
        print("\nPostgreSQL connection closed.")


async def demo_api_server_pattern():
    """Demonstrate the API server pattern for multi-turn conversations.

    This shows how to structure an API endpoint that maintains
    conversation state across requests.
    """
    from mask.checkpointer import create_postgres_checkpointer, create_thread_config

    postgres_url = os.environ.get(
        "POSTGRES_URL",
        "postgresql://postgres:postgres@localhost:5432/mask_db"
    )

    # In a FastAPI/Starlette app, you'd typically:
    # 1. Create checkpointer at startup
    # 2. Use dependency injection to access it in routes
    # 3. Close at shutdown

    print("API Server Pattern Example")
    print("-" * 40)
    print()

    # Startup
    checkpointer = await create_postgres_checkpointer(postgres_url)

    try:
        # Simulate API requests
        async def handle_chat_request(user_id: str, session_id: str, message: str):
            """Handle a chat request (simulated API endpoint)."""
            # Create thread ID from user and session
            thread_id = f"{user_id}-{session_id}"
            config = create_thread_config(thread_id)

            # In real app:
            # response = await graph.ainvoke(
            #     {"messages": [HumanMessage(content=message)]},
            #     config=config,
            # )
            # return response["messages"][-1].content

            return f"Response to: {message} (thread: {thread_id})"

        # Simulate requests from same user/session
        print("Request 1:")
        r1 = await handle_chat_request("user-123", "session-456", "Hello!")
        print(f"  Response: {r1}")

        print("\nRequest 2 (same session):")
        r2 = await handle_chat_request("user-123", "session-456", "Remember my name is Bob")
        print(f"  Response: {r2}")

        print("\nRequest 3 (same session):")
        r3 = await handle_chat_request("user-123", "session-456", "What's my name?")
        print(f"  Response: {r3}")

        print("\nRequest 4 (different session):")
        r4 = await handle_chat_request("user-123", "session-789", "Hello!")
        print(f"  Response: {r4}")

    finally:
        # Shutdown
        await checkpointer.close()


if __name__ == "__main__":
    # Run main demo
    asyncio.run(main())

    print("\n" + "=" * 60 + "\n")

    # Run API pattern demo
    asyncio.run(demo_api_server_pattern())
