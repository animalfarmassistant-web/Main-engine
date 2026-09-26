"""Spoken conversation about the codebase (``strix vibe chat``).

Unlike ``PromptSynthesizer``, which writes a long Markdown artifact meant to be
read, this holds a back-and-forth dialogue whose replies are meant to be *heard*:
short, plain sentences with no Markdown, no code blocks, and no path soup.

Each turn retrieves fresh graph context for what the user just said, so answers
stay grounded in the actual repository rather than the model's imagination.
Conversation history is kept (and trimmed) so follow-ups like "what about the
tests for it?" resolve against the previous turns.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from strix.vibe.llm import run_completion


if TYPE_CHECKING:
    from strix.vibe.config import TargetTool, VibeSettings
    from strix.vibe.graph import GraphifyManager


logger = logging.getLogger(__name__)

# Keep the tail of the dialogue; older turns fall out of the window.
_MAX_HISTORY_TURNS = 12
_MAX_CONTEXT_NODES = 8

_CHAT_SYSTEM = (
    "You are a voice assistant that answers questions about a specific software "
    "project. Your replies are spoken aloud, so they must be:\n"
    "- SHORT: at most 3 sentences unless the user explicitly asks for detail.\n"
    "- Plain spoken English: no Markdown, no bullet points, no code blocks, no "
    "URLs, and no long file paths read out character by character. Say "
    "'the codex config module' rather than 'strix/config/codex.py'.\n"
    "- Grounded: use the Context section supplied with each question. If the "
    "context does not answer it, say so plainly and suggest what to look at "
    "next. Never invent file names, functions, or behaviour.\n"
    "- Conversational: the user is talking to you, so answer the question that "
    "was asked and stop. Do not summarise the whole project unprompted.\n"
    "If the user asks you to write or generate a prompt, tell them to say "
    "'generate a prompt for that' and confirm briefly what it will cover."
)


# Reasoning models (e.g. Gemini 3) prepend their deliberation to the answer,
# separated by a run of blank lines: "**My Thoughts**\n\nOkay, so...\n\n\n\n<answer>".
# Speaking that aloud is unusable, so keep only the trailing answer.
_REASONING_SPLIT_RE = re.compile(r"\n\s*\n\s*\n+")
_THINKING_OPENER_RE = re.compile(
    r"^\s*(\*\*[^\n]{0,80}\*\*\s*\n"
    r"|okay[,;: ]|alright[,;: ]|let me think|so,? the (user|question))",
    re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """Drop a reasoning model's thinking preamble, keeping the spoken answer.

    Only trims when the leading block actually looks like deliberation, so a
    genuine multi-paragraph answer survives intact.
    """
    cleaned = text.strip()
    chunks = [chunk.strip() for chunk in _REASONING_SPLIT_RE.split(cleaned) if chunk.strip()]
    if len(chunks) < 2:
        return cleaned
    if _THINKING_OPENER_RE.match(chunks[0]):
        return chunks[-1]
    return cleaned


def format_context(nodes: list[dict[str, Any]]) -> str:
    """Render retrieved graph nodes as a compact context block for the model."""
    if not nodes:
        return "(no matching code was found in the project index for this question)"
    lines = []
    for node in nodes[:_MAX_CONTEXT_NODES]:
        label = node.get("label", "?")
        kind = node.get("kind", "unknown")
        source = node.get("source_file") or "unknown file"
        lines.append(f"- {label} ({kind}) in {source}")
    return "\n".join(lines)


class ChatSession:
    """A stateful, graph-grounded voice conversation."""

    def __init__(self, settings: VibeSettings, manager: GraphifyManager) -> None:
        self._settings = settings
        self._manager = manager
        self._history: list[dict[str, str]] = []

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    def _trim(self) -> None:
        if len(self._history) > _MAX_HISTORY_TURNS * 2:
            self._history = self._history[-_MAX_HISTORY_TURNS * 2 :]

    def build_turn_input(self, utterance: str, context: str) -> str:
        """Compose the user-side message: prior dialogue + fresh context + question."""
        parts: list[str] = []
        if self._history:
            transcript = "\n".join(
                f"{'User' if m['role'] == 'user' else 'You'}: {m['content']}"
                for m in self._history
            )
            parts.append(f"Conversation so far:\n{transcript}")
        parts.append(f"Context from the project index:\n{context}")
        parts.append(f"User just said: {utterance}")
        return "\n\n".join(parts)

    async def ask(self, utterance: str) -> str:
        """Answer one spoken turn, grounded in freshly retrieved graph context."""
        utterance = utterance.strip()
        if not utterance:
            return ""
        nodes = self._manager.query(utterance)
        raw = await run_completion(
            _CHAT_SYSTEM,
            self.build_turn_input(utterance, format_context(nodes)),
            # Spoken answers are short; a reasoning model's deliberation only
            # adds latency and leaks into the reply.
            reasoning_effort="none",
        )
        reply = strip_reasoning(raw)
        self._history.append({"role": "user", "content": utterance})
        self._history.append({"role": "assistant", "content": reply})
        self._trim()
        return reply

    async def generate(self, tool: TargetTool) -> str:
        """Author a detailed engineering prompt from the whole conversation so far.

        Re-queries the graph against the full transcript (turn-level context
        isn't kept in ``history``, only the utterance/reply text) to rebuild a
        context brief, then reuses the exact code-change synthesis path used by
        ``strix vibe generate``.
        """
        from strix.vibe.builder import PromptBuilder
        from strix.vibe.synthesizer import PromptSynthesizer

        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in self._history)
        if not transcript.strip():
            raise RuntimeError("nothing has been discussed yet - ask a question first")
        nodes = self._manager.query(transcript)
        brief = PromptBuilder(self._settings).build(
            intent=transcript, tool=tool, context_nodes=nodes
        )
        return await PromptSynthesizer(self._settings).synthesize(
            intent=transcript, context_brief=brief, tool=tool
        )
