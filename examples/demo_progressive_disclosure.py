"""Demo script for Progressive Disclosure.

This script demonstrates how Progressive Disclosure works in MASK:
1. Initially, only loader tools are visible
2. When a skill is activated, its capability tools become available
3. Instructions are injected into the system prompt

Run with:
    python examples/demo_progressive_disclosure.py
"""

import asyncio
from pathlib import Path

from mask.core import SkillRegistry
from mask.middleware import SkillMiddleware


def main():
    """Demonstrate Progressive Disclosure."""
    print("=" * 60)
    print("MASK Progressive Disclosure Demo")
    print("=" * 60)

    # Initialize registry and discover skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"

    print(f"\nDiscovering skills from: {skills_dir}")
    count = registry.discover_from_directory(skills_dir)
    print(f"Found {count} skills")

    # List all skills
    print("\n--- Registered Skills ---")
    for skill_info in registry.get_skills_summary():
        print(f"  - {skill_info['name']}: {skill_info['description']}")

    # Phase 1: No skills activated
    print("\n--- Phase 1: No Skills Activated ---")
    tools = registry.get_tools_for_active_skills([])
    print(f"Available tools ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")

    # Phase 2: Activate pdf-processing
    print("\n--- Phase 2: Activate pdf-processing ---")
    active_skills = ["pdf-processing"]
    tools = registry.get_tools_for_active_skills(active_skills)
    print(f"Available tools ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool.name}")

    # Show middleware behavior
    print("\n--- Middleware Demo ---")
    middleware = SkillMiddleware(registry)

    # Build state with no skills
    state = {"messages": [], "skills_loaded": []}
    tools = middleware.get_tools(state)
    print(f"Tools with no active skills: {len(tools)}")

    # Build state with pdf-processing active
    state = {"messages": [], "skills_loaded": ["pdf-processing"]}
    tools = middleware.get_tools(state)
    print(f"Tools with pdf-processing active: {len(tools)}")

    # Show system prompt injection
    print("\n--- System Prompt Injection ---")
    from mask.middleware.skill_middleware import build_skills_system_prompt

    prompt = build_skills_system_prompt(registry, ["pdf-processing"])
    print("Generated system prompt section:")
    print("-" * 40)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
