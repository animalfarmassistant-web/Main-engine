"""Settings for the ``vibe`` read-only prompt orchestrator.

``VibeSettings`` covers only vibe-specific knobs (where to read, where to write,
which external tool to target). The LLM configuration is shared with the rest of
Strix — reach it via :func:`strix.config.load_settings().llm` — so a single
``STRIX_LLM``/``OLLAMA_API_BASE`` setup drives both scans and prompt polishing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


TargetTool = Literal["claude", "chatgpt", "cursor", "generic"]

_GRAPHIFY_OUT_DIRNAME = "graphify-out"


class VibeSettings(BaseSettings):
    """Read/write locations and defaults for the vibe prompt orchestrator."""

    model_config = SettingsConfigDict(
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    project_root: Path = Field(default_factory=Path.cwd, alias="VIBE_PROJECT_ROOT")
    out_dirname: str = Field(default=".vibe", alias="VIBE_OUT_DIR")
    default_tool: TargetTool = Field(default="claude", alias="VIBE_DEFAULT_TOOL")
    # Speech-to-text model, kept separate from STRIX_LLM: not every chat model
    # accepts audio input, so voice transcription can differ from synthesis.
    # Verified live: gemini-3-flash-preview accepts audio AND returns empty for
    # non-speech. (gemini-3.1-flash-lite accepts audio but hallucinates words
    # from noise; gemini-2.0-flash is retired despite LiteLLM's stale
    # supports_audio_input flag.)
    stt_model: str = Field(default="gemini/gemini-3-flash-preview", alias="VIBE_STT_MODEL")

    def model_post_init(self, _context: object) -> None:
        # Normalise so every derived path is absolute and comparisons in the
        # write-guard are stable.
        object.__setattr__(self, "project_root", self.project_root.expanduser().resolve())

    @property
    def out_dir(self) -> Path:
        """The vibe state dir, e.g. ``<project>/.vibe``."""
        return self.project_root / self.out_dirname

    @property
    def prompts_dir(self) -> Path:
        """Where generated prompt briefs are written."""
        return self.out_dir / "prompts"

    @property
    def memory_path(self) -> Path:
        """The standalone vibe memory store."""
        return self.out_dir / "memory.json"

    @property
    def graph_out_dir(self) -> Path:
        """Graphify's artifact dir, kept under ``.vibe`` so the source tree is untouched."""
        return self.out_dir / _GRAPHIFY_OUT_DIRNAME

    @property
    def graph_json_path(self) -> Path:
        return self.graph_out_dir / "graph.json"

    @property
    def allowed_write_roots(self) -> tuple[Path, ...]:
        """Roots the write-guard permits. Only the vibe state dir — which already
        contains the graphify-out subdir — so nothing lands in the project."""
        return (self.out_dir,)
