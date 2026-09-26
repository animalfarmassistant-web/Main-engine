"""Prompt synthesizer for the vibe orchestrator.

Where ``PromptBuilder`` assembles a deterministic *context brief* and ``polish``
only tightens it, the synthesizer *authors* a fully articulated engineering
prompt: it takes the context brief plus the user's raw intent and asks the LLM
to produce explicit implementation steps, exact file targets, edge cases, and
acceptance criteria — the ready-to-paste prompt for an external AI coding tool.

Unlike polish (best-effort), synthesis requires a model; the CLI surfaces a
clear error if none is configured.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from strix.vibe.llm import run_completion


if TYPE_CHECKING:
    from strix.vibe.config import TargetTool, VibeSettings


_SYNTHESIZER_SYSTEM = (
    "You are an expert prompt engineer and senior software architect. You are "
    "given a structured Context Map of a codebase (relevant files, symbols, their "
    "relationships, and notes from prior sessions) and a user's raw feature "
    "request. Produce a single, fully articulated, production-ready engineering "
    "prompt for an AI coding tool.\n\n"
    "The prompt you write MUST contain, as Markdown sections:\n"
    "1. Objective — one or two sentences.\n"
    "2. Files to modify — use ONLY paths that appear in the Context Map; for each, "
    "say what changes there.\n"
    "3. Implementation steps — explicit, ordered, and concrete.\n"
    "4. Edge cases & failure modes to handle.\n"
    "5. Acceptance criteria — including the tests that must pass.\n"
    "6. Unknowns to confirm — anything the Context Map does not cover that the "
    "implementer must check first (never invent file paths or APIs to fill a gap).\n\n"
    "Ground every instruction in the Context Map. Output ONLY the finished prompt "
    "in Markdown — no preamble, no commentary about your process."
)

# Anchors for trimming a reasoning model's "thinking" preamble: prefer the
# mandated first section, else the first Markdown heading of any level.
_OBJECTIVE_HEADING_RE = re.compile(r"(?im)^\s{0,3}#{1,6}\s+.*\bobjective\b")
_ANY_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S")


def strip_thinking_preamble(text: str) -> str:
    """Drop any leading 'thinking' prose a reasoning model (e.g. Gemini 3) emits
    before the actual prompt.

    Anchors on the first Markdown heading (preferring one containing
    "objective", since that's what the code-change synthesis system prompt
    mandates as its first section — other Markdown-headed syntheses, like the
    planning-mode artifact, just fall through to the first heading of any
    kind). If no heading is present, the text is returned unchanged rather
    than risk mangling a differently-shaped reply. Shared with
    ``strix.vibe.planning``, whose synthesized output is Markdown-headed the
    same way.
    """
    match = _OBJECTIVE_HEADING_RE.search(text) or _ANY_HEADING_RE.search(text)
    if match is None:
        return text
    return text[match.start() :].lstrip()


class PromptSynthesizer:
    """Turns a context brief + raw intent into a detailed engineering prompt."""

    def __init__(self, settings: VibeSettings) -> None:
        self._settings = settings

    def build_user_message(self, *, intent: str, context_brief: str, tool: TargetTool) -> str:
        return (
            f"# Target AI coding tool\n{tool}\n\n"
            f"# User feature request (raw intent)\n{intent.strip()}\n\n"
            f"# Context Map of the codebase\n{context_brief.strip()}\n"
        )

    async def synthesize(
        self, *, intent: str, context_brief: str, tool: TargetTool
    ) -> str:
        """Author the detailed prompt. Raises ``NoModelConfiguredError`` if no LLM;
        propagates provider errors so the CLI can report a hard failure."""
        user_message = self.build_user_message(
            intent=intent, context_brief=context_brief, tool=tool
        )
        result = await run_completion(_SYNTHESIZER_SYSTEM, user_message)
        if not result:
            raise RuntimeError("the model returned an empty synthesis")
        return strip_thinking_preamble(result)
