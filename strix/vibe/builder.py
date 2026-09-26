"""Prompt assembly for the vibe orchestrator.

Two stages, both optional-safe:

1. **Deterministic render** — a Jinja template turns the retrieved project
   context + memory into a structured, copy-pasteable prompt. Works fully
   offline, no model required. This is always the source of truth.
2. **Optional polish** — when the user passes ``--polish``, the assembled draft
   is handed to the configured (typically local, e.g. Ollama) LLM for a single
   tightening pass, reusing Strix's model plumbing. Any failure returns the
   deterministic draft unchanged, so polish can never break output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


if TYPE_CHECKING:
    from strix.vibe.config import TargetTool, VibeSettings


logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "prompt_brief.jinja"

_TOOL_LABELS = {
    "claude": "Claude Code",
    "chatgpt": "ChatGPT",
    "cursor": "Cursor",
    "generic": "an AI coding assistant",
}
_ROLE_TEXT = {
    "claude": (
        "Use your tools to read the referenced files before editing. Make the "
        "smallest change that fully satisfies the task, and match the "
        "surrounding code's style."
    ),
    "cursor": (
        "Work within the open workspace. Read the referenced files for context "
        "before proposing edits, and keep changes tightly scoped."
    ),
    "chatgpt": (
        "You do not have the repository open, so rely on the context map below. "
        "When you need a file that isn't shown, say so explicitly instead of "
        "guessing."
    ),
    "generic": "Ground every change in the context map below; state any assumption you must make.",
}
_OUTPUT_TEXT = {
    "chatgpt": (
        "Return each changed file as a fenced code block labelled with its path, "
        "preceded by a one-line rationale. End with a short list of anything you "
        "assumed or could not verify."
    ),
}
_OUTPUT_DEFAULT = (
    "Apply the edits directly, then summarise what changed and why, and list any "
    "follow-ups or assumptions."
)
_BASE_CONSTRAINTS = (
    "- Stay within the scope of the objective above; do not refactor unrelated code.",
    "- Preserve existing public interfaces unless the objective requires changing them.",
    (
        "- Follow the conventions already present in the referenced files "
        "(naming, error handling, tests)."
    ),
)

_POLISH_SYSTEM = (
    "You are a prompt engineer. You are given a task brief that will be pasted "
    "into an external AI coding tool. Tighten it: remove redundancy, sharpen the "
    "objective and acceptance criteria, and keep every file path, symbol name, "
    "and section heading exactly as given. Do not invent files or facts that are "
    "not already in the brief. Return only the improved brief in Markdown."
)


class PromptBuilder:
    """Assembles prompt briefs from project context and optionally polishes them."""

    def __init__(self, settings: VibeSettings) -> None:
        self._settings = settings
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(enabled_extensions=(), default_for_string=False),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def build(
        self,
        *,
        intent: str,
        tool: TargetTool,
        context_nodes: list[dict[str, Any]],
        memory_notes: list[dict[str, Any]] | None = None,
        extra_constraints: list[str] | None = None,
    ) -> str:
        """Render the deterministic prompt brief (no LLM)."""
        files: list[str] = []
        for node in context_nodes:
            src = node.get("source_file")
            if src and src not in files:
                files.append(src)
        file_lines = [f"- `{f}`" for f in files]
        symbol_lines = [_format_symbol_line(node) for node in context_nodes]
        memory_lines = [_format_memory_line(note) for note in (memory_notes or [])]
        constraint_lines = [*_BASE_CONSTRAINTS, *(f"- {c}" for c in (extra_constraints or []))]

        template = self._env.get_template(_TEMPLATE_NAME)
        return template.render(
            intent=intent.strip(),
            tool_label=_TOOL_LABELS.get(tool, _TOOL_LABELS["generic"]),
            role_text=_ROLE_TEXT.get(tool, _ROLE_TEXT["generic"]),
            output_text=_OUTPUT_TEXT.get(tool, _OUTPUT_DEFAULT),
            project_name=self._settings.project_root.name or "this project",
            file_lines=file_lines,
            symbol_lines=symbol_lines,
            memory_lines=memory_lines,
            constraint_lines=constraint_lines,
        )

    async def polish(self, draft: str) -> str:
        """Run one LLM tightening pass over *draft*; return the draft on any failure."""
        from strix.vibe.llm import run_completion

        try:
            polished = await run_completion(_POLISH_SYSTEM, draft)
        except Exception:
            logger.warning("prompt polish failed; returning the deterministic draft", exc_info=True)
            return draft
        return polished or draft


def _format_symbol_line(node: dict[str, Any]) -> str:
    parts = [f"- **{node.get('label', node.get('id', '?'))}** ({node.get('kind', 'unknown')})"]
    if node.get("source_file"):
        parts.append(f" - `{node['source_file']}`")
    neighbours = node.get("neighbours") or []
    if neighbours:
        parts.append(f" | related: {', '.join(str(n) for n in neighbours)}")
    return "".join(parts)


def _format_memory_line(note: dict[str, Any]) -> str:
    category = note.get("category", "general")
    title = note.get("title", "")
    content = note.get("content", "")
    return f"- _{category}_ **{title}**: {content}"
