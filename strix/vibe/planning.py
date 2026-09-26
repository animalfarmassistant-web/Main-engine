"""Planning Phase: interview a user about a project that doesn't exist yet.

``ChatSession`` (in ``conversation.py``) assumes there is already code to ground
answers in. When ``GraphifyManager.has_source_files()`` is False, ``vibe chat``
switches to ``PlanningSession`` instead: rather than answering questions about a
codebase, it *asks* the user questions to build up a project concept, remembers
what it learns (as ``plan``-category ``VibeMemory`` entries, so the concept
survives across process restarts), and on request synthesizes everything
discussed into an implementation-plan prompt to paste into an AI coding tool.

Same ``ask(utterance) -> str`` interface as ``ChatSession`` so the ``vibe chat``
loop in ``cli.py`` is mode-agnostic.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from strix.vibe.conversation import strip_reasoning
from strix.vibe.llm import run_completion
from strix.vibe.memory import VibeMemory
from strix.vibe.synthesizer import strip_thinking_preamble


if TYPE_CHECKING:
    from strix.vibe.config import TargetTool, VibeSettings


logger = logging.getLogger(__name__)

_MAX_HISTORY_TURNS = 12
_MAX_PLAN_NOTES = 20

_PLANNING_SYSTEM = (
    "You are a voice assistant helping a user think through a NEW software "
    "project that does not have any code yet - there is nothing to analyze, "
    "so never claim to be reading files or describe yourself as part of an "
    "existing codebase. Your job is to interview the user about their idea.\n\n"
    "Your replies are spoken aloud, so they must be:\n"
    "- SHORT: at most 2-3 sentences.\n"
    "- Plain spoken English: no Markdown, no bullet points, no code blocks.\n"
    "- ONE focused question per turn, picking whichever of these is least "
    "clear so far: what the project actually does, who it's for, its core "
    "features, any constraints (platform, offline use, must-have tech), and "
    "what 'done' looks like for a first version. Don't ask about everything "
    "at once.\n"
    "- Grounded only in what the user has actually said: never invent "
    "features, users, or tech choices they haven't mentioned.\n"
    "If the user asks you to write or generate a prompt or plan, tell them to "
    "say 'generate a prompt for that' and confirm briefly what it will cover."
)

_PLANNING_SYNTHESIS_SYSTEM = (
    "You are an expert software architect. You are given notes and a "
    "transcript from an interview about a NEW software project that has no "
    "code yet. Produce a single, fully articulated implementation-plan prompt "
    "for an AI coding tool to scaffold the project from scratch.\n\n"
    "The prompt you write MUST contain, as Markdown sections:\n"
    "1. Concept & Goals - what the project is and why it exists, from what "
    "was actually discussed.\n"
    "2. Proposed Features - concrete, based on the discussion.\n"
    "3. Suggested Tech Stack - only what was actually discussed; write "
    "'to be decided' for anything that wasn't.\n"
    "4. Implementation Milestones - an ordered, incremental build plan.\n"
    "5. Open Questions - anything still unresolved that the user or "
    "implementer must decide before or during the build.\n\n"
    "Never invent features, users, or technical decisions that were not part "
    "of the discussion; when something is unknown, say so in Open Questions "
    "instead of guessing. Output ONLY the finished prompt in Markdown - no "
    "preamble, no commentary about your process."
)


class PlanningSession:
    """Interview-driven conversation that builds a project concept from scratch."""

    def __init__(self, settings: VibeSettings) -> None:
        self._settings = settings
        self._memory = VibeMemory(settings)
        self._history: list[dict[str, str]] = []

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def _trim(self) -> None:
        if len(self._history) > _MAX_HISTORY_TURNS * 2:
            self._history = self._history[-_MAX_HISTORY_TURNS * 2 :]

    def _build_turn_input(self, utterance: str) -> str:
        parts: list[str] = []
        if self._history:
            transcript = "\n".join(
                f"{'User' if m['role'] == 'user' else 'You'}: {m['content']}"
                for m in self._history
            )
            parts.append(f"Conversation so far:\n{transcript}")
        parts.append(f"User just said: {utterance}")
        return "\n\n".join(parts)

    async def ask(self, utterance: str) -> str:
        """Answer (and ask a follow-up about) one interview turn."""
        utterance = utterance.strip()
        if not utterance:
            return ""
        raw = await run_completion(
            _PLANNING_SYSTEM,
            self._build_turn_input(utterance),
            reasoning_effort="none",
        )
        reply = strip_reasoning(raw)
        self._history.append({"role": "user", "content": utterance})
        self._history.append({"role": "assistant", "content": reply})
        self._trim()
        self._remember(utterance, reply)
        return reply

    def _remember(self, utterance: str, reply: str) -> None:
        """Persist what was learned this turn, so it survives a process restart."""
        try:
            self._memory.add(
                title=f"Planning: {utterance[:80]}",
                content=f"User said: {utterance}\nAssistant: {reply}",
                category="plan",
                tags=["planning-interview"],
            )
        except Exception:  # memory is a convenience here, not a hard dependency.
            logger.debug("could not persist planning memory entry", exc_info=True)

    async def generate(self, tool: TargetTool) -> str:
        """Synthesize everything discussed into an implementation-plan prompt."""
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in self._history)
        notes = self._memory.list(category="plan")[:_MAX_PLAN_NOTES]
        notes_block = "\n".join(f"- {n['title']}: {n['content']}" for n in notes)
        if not transcript.strip() and not notes_block.strip():
            raise RuntimeError("nothing has been discussed yet - talk about the project first")

        user_message = (
            f"# Conversation transcript\n{transcript or '(none this session)'}\n\n"
            f"# Saved planning notes from this and prior sessions\n"
            f"{notes_block or '(none)'}\n\n"
            f"# Target AI coding tool\n{tool}\n"
        )
        result = await run_completion(_PLANNING_SYNTHESIS_SYSTEM, user_message)
        if not result:
            raise RuntimeError("the model returned an empty plan")
        return strip_thinking_preamble(result)
