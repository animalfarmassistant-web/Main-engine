"""Standalone project memory for the vibe orchestrator (``.vibe/memory.json``).

This mirrors the *shape* of Strix's in-run notes store (category + tags +
timestamps) and its atomic-write pattern, but is deliberately **independent**:
it does not touch the notes module's process-global ``_notes_storage`` /
``_notes_path`` (which hardcode ``notes.json`` and would bleed state across the
process). VibeMemory owns its own file and never writes outside the vibe
workspace (enforced via :func:`strix.vibe.paths.assert_write_allowed`).

Memory tracks feature progress, decisions, and architectural state across
sessions so generated prompts stay grounded in what has already been done.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from strix.vibe.paths import safe_write_text


if TYPE_CHECKING:
    from pathlib import Path

    from strix.vibe.config import VibeSettings


logger = logging.getLogger(__name__)

VALID_CATEGORIES = (
    "general",
    "plan",
    "progress",
    "findings",
    "methodology",
    "decision",
)
_ID_LEN = 6
_ID_ATTEMPTS = 1024


class VibeMemory:
    """A tiny JSON-backed note store scoped to one project's ``.vibe/memory.json``."""

    def __init__(self, settings: VibeSettings) -> None:
        self._settings = settings
        self._path: Path = settings.memory_path
        self._allowed = settings.allowed_write_roots
        self._store: dict[str, dict[str, Any]] = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        self._store = {}
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("vibe memory at %s is unreadable; starting empty", self._path)
            return
        if isinstance(data, dict):
            self._store = {
                nid: note
                for nid, note in data.items()
                if isinstance(nid, str) and isinstance(note, dict)
            }

    def _persist(self) -> None:
        payload = json.dumps(self._store, ensure_ascii=False, indent=2, default=str)
        safe_write_text(self._path, payload, self._allowed)

    def _new_id(self) -> str:
        for _ in range(_ID_ATTEMPTS):
            candidate = uuid.uuid4().hex[:_ID_LEN]
            if candidate not in self._store:
                return candidate
        raise RuntimeError("could not allocate a unique memory id")

    # -- public API --------------------------------------------------------
    def add(
        self,
        title: str,
        content: str,
        *,
        category: str = "general",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a memory entry and persist. Returns the stored entry with ``id``."""
        if not title.strip():
            raise ValueError("memory title cannot be empty")
        if not content.strip():
            raise ValueError("memory content cannot be empty")
        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"invalid category {category!r}; must be one of: {', '.join(VALID_CATEGORIES)}"
            )
        now = datetime.now(UTC).isoformat()
        note_id = self._new_id()
        note = {
            "title": title.strip(),
            "content": content.strip(),
            "category": category,
            "tags": tags or [],
            "created_at": now,
            "updated_at": now,
        }
        self._store[note_id] = note
        self._persist()
        return {"id": note_id, **note}

    def get(self, note_id: str) -> dict[str, Any] | None:
        note = self._store.get(note_id)
        return None if note is None else {"id": note_id, **note}

    def list(
        self,
        *,
        category: str | None = None,
        tags: list[str] | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return matching entries, newest first. Filters compose (AND)."""
        results: list[dict[str, Any]] = []
        needle = search.lower() if search else None
        for note_id, note in self._store.items():
            if category and note.get("category") != category:
                continue
            if tags and not (set(tags) & set(note.get("tags", []))):
                continue
            if needle and needle not in note.get("title", "").lower() and (
                needle not in note.get("content", "").lower()
            ):
                continue
            results.append({"id": note_id, **note})
        results.sort(key=lambda n: n.get("created_at", ""), reverse=True)
        return results
