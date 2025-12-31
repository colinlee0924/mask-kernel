"""MASK CLI.

This module provides the command-line interface for MASK Kernel.

Commands:
- mask init: Create a new agent project
- mask run: Run an agent interactively or as server
"""

from mask.cli.main import app, main

__all__ = ["app", "main"]
