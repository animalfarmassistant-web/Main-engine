"""Read-only project analysis + external-prompt generation (``strix vibe``).

This subpackage is intentionally decoupled from the Strix scan engine: it never
imports ``strix.runtime`` (no Docker) and never writes inside the user's project.
It indexes a codebase into a knowledge graph (vendored Graphify, with a pure
``ast`` fallback), tracks state in a local memory store, and assembles optimized
prompts for external AI tools.
"""

from __future__ import annotations


__all__ = ["run_vibe"]


def run_vibe(argv: list[str]) -> int:
    """Lazy proxy to :func:`strix.vibe.cli.run_vibe` (keeps import cost off cold paths)."""
    from strix.vibe.cli import run_vibe as _run_vibe

    return _run_vibe(argv)
