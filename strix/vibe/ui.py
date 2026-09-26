"""Interactive Textual UI for the vibe orchestrator (``strix vibe ui``).

A lightweight, self-contained terminal app — deliberately NOT built on the Go
scan sidecar in ``strix/interface/tui`` (that is wired to the pentest engine).
Workflow:

- Type an intent in the input box; the project graph is searched live
  (debounced) and matching files/symbols appear in the table.
- F2 (or Ctrl+G) synthesizes a detailed engineering prompt via the LLM and
  shows it in the output pane (also saved under ``.vibe/prompts``).
- F3 (or Ctrl+Y) copies the generated prompt to the clipboard.

``textual`` is an optional dependency (installed via ``pip install .[vibe]``);
the CLI reports a clear hint if it is missing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Input, Label, TextArea

from strix.vibe.builder import PromptBuilder
from strix.vibe.cli import _copy_to_clipboard, _slugify
from strix.vibe.graph import GraphifyManager
from strix.vibe.llm import NoModelConfiguredError
from strix.vibe.memory import VibeMemory
from strix.vibe.paths import safe_write_text
from strix.vibe.synthesizer import PromptSynthesizer


if TYPE_CHECKING:
    from textual.timer import Timer

    from strix.vibe.config import TargetTool, VibeSettings
    from strix.vibe.voice import Recorder


logger = logging.getLogger(__name__)

_SEARCH_DEBOUNCE_S = 0.3
_MAX_MEMORY_NOTES = 8


class VibeUIApp(App[None]):
    """Interactive intent → context → synthesized-prompt loop."""

    TITLE = "vibe"
    SUB_TITLE = "read-only prompt orchestrator"

    CSS = """
    #status { height: 1; padding: 0 1; background: $panel; color: $text-muted; }
    #matches { height: 40%; border-bottom: solid $accent; }
    #output { height: 1fr; }
    """

    # Function-key aliases matter: terminal hosts (notably VS Code's integrated
    # terminal) grab ctrl+g / ctrl+y / ctrl+r for their own commands, so the
    # keypress never reaches this app. F2/F3/F4 pass through everywhere.
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("f2", "generate", "Generate (F2/^g)"),
        Binding("ctrl+g", "generate", "Generate", show=False),
        Binding("f3", "copy", "Copy (F3/^y)"),
        Binding("ctrl+y", "copy", "Copy", show=False),
        Binding("f4", "reindex", "Re-index (F4/^r)"),
        Binding("ctrl+r", "reindex", "Re-index", show=False),
        Binding("f6", "voice", "Voice (F6)"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, settings: VibeSettings, tool: TargetTool) -> None:
        super().__init__()
        self._settings = settings
        self._tool = tool
        self._manager = GraphifyManager(settings)
        self._memory = VibeMemory(settings)
        self._builder = PromptBuilder(settings)
        self._synthesizer = PromptSynthesizer(settings)
        self._debounce: Timer | None = None
        self._pending_intent = ""
        self._generated = ""
        self._recorder: Recorder | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(
            placeholder="Describe the change - typing/Enter searches, F2 generates",
            id="intent",
        )
        yield Label("Starting...", id="status")
        table: DataTable[str] = DataTable(id="matches", cursor_type="row", zebra_stripes=True)
        table.add_columns("Kind", "Symbol", "File")
        yield table
        yield TextArea("", id="output", read_only=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#intent", Input).focus()
        self._index(force=False)

    # -- status helper -----------------------------------------------------
    def _status(self, message: str) -> None:
        self.query_one("#status", Label).update(message)

    # -- indexing ----------------------------------------------------------
    @work(exclusive=True, group="index")
    async def _index(self, *, force: bool) -> None:
        self._status("Indexing project... (first run can take a while)")
        try:
            await asyncio.to_thread(self._manager.ensure_index, force=force)
        except Exception as exc:
            logger.exception("vibe ui indexing failed")
            self._status(f"Index error: {exc}")
            return
        node_count = len(self._manager.load_graph().get("nodes", []))
        self._status(f"Indexed {node_count} nodes. Type an intent to search; press F2 to generate.")

    def action_reindex(self) -> None:
        self._index(force=True)

    # -- live search -------------------------------------------------------
    @on(Input.Changed, "#intent")
    def _on_intent_changed(self, event: Input.Changed) -> None:
        self._pending_intent = event.value
        if self._debounce is not None:
            self._debounce.stop()
        self._debounce = self.set_timer(_SEARCH_DEBOUNCE_S, self._run_search)

    def _run_search(self) -> None:
        self._search(self._pending_intent)

    @on(Input.Submitted, "#intent")
    def _on_intent_submitted(self, event: Input.Submitted) -> None:
        # Enter runs the search immediately (bypass the debounce).
        if self._debounce is not None:
            self._debounce.stop()
        self._search(event.value)

    @work(exclusive=True, group="search")
    async def _search(self, intent: str) -> None:
        table = self.query_one("#matches", DataTable)
        table.clear()
        intent = intent.strip()
        if not intent:
            return
        hits = await asyncio.to_thread(self._manager.query, intent)
        for hit in hits:
            table.add_row(
                str(hit.get("kind", "?")),
                str(hit.get("label", "")),
                str(hit.get("source_file") or ""),
            )
        self._status(
            f"{len(hits)} match(es) for '{intent[:48]}'. Press F2 to generate."
            if hits
            else f"No matches for '{intent[:48]}'. Try different words, or press F2 anyway."
        )

    # -- voice -------------------------------------------------------------
    def action_voice(self) -> None:
        """F6 toggles recording: first press starts, second stops + transcribes."""
        from strix.vibe.voice import Recorder, VoiceUnavailableError

        if self._recorder is not None and self._recorder.is_recording:
            recorder, self._recorder = self._recorder, None
            self._transcribe(recorder.stop())
            return
        recorder = Recorder()
        try:
            recorder.start()
        except VoiceUnavailableError as exc:
            self._status("Microphone unavailable.")
            self.notify(str(exc), severity="error", timeout=10)
            return
        self._recorder = recorder
        self._status("Recording... press F6 again to stop.")

    @work(exclusive=True, group="voice")
    async def _transcribe(self, wav_bytes: bytes) -> None:
        from strix.vibe.voice import transcribe

        self._status("Transcribing...")
        try:
            text = await transcribe(wav_bytes, self._settings)
        except Exception as exc:
            logger.exception("vibe ui transcription failed")
            self._status("Transcription failed.")
            self.notify(f"Transcription failed: {exc}", severity="error", timeout=10)
            return
        if not text:
            self._status("Nothing transcribed - microphone silent?")
            return
        # Setting the value fires Input.Changed, which runs the debounced search.
        self.query_one("#intent", Input).value = text

    # -- generate ----------------------------------------------------------
    def action_generate(self) -> None:
        intent = self.query_one("#intent", Input).value.strip()
        if not intent:
            self.notify("Enter an intent first.", severity="warning")
            return
        self._generate(intent)

    @work(exclusive=True, group="generate")
    async def _generate(self, intent: str) -> None:
        self._status("Generating via LLM... (this can take ~15s)")
        hits = await asyncio.to_thread(self._manager.query, intent)
        notes = self._memory.list()[:_MAX_MEMORY_NOTES]
        brief = self._builder.build(
            intent=intent, tool=self._tool, context_nodes=hits, memory_notes=notes
        )
        try:
            detailed = await self._synthesizer.synthesize(
                intent=intent, context_brief=brief, tool=self._tool
            )
        except NoModelConfiguredError as exc:
            self._status("No LLM configured.")
            self.notify(str(exc), severity="error", timeout=10)
            return
        except Exception as exc:
            logger.exception("vibe ui generation failed")
            self._status("Generation failed.")
            self.notify(f"Generation failed: {exc}", severity="error", timeout=10)
            return

        self._generated = detailed
        self.query_one("#output", TextArea).text = detailed
        self._save_and_record(intent, detailed, len(hits))

    def _save_and_record(self, intent: str, text: str, context_len: int) -> None:
        out_path = self._settings.prompts_dir / f"{_slugify(intent)}.generated.md"
        try:
            safe_write_text(out_path, text, self._settings.allowed_write_roots)
            self._status(f"Saved to {out_path}  (F3 to copy)")
        except Exception:
            logger.debug("vibe ui could not save prompt", exc_info=True)
            self._status("Generated (F3 to copy).")
        try:
            self._memory.add(
                title=f"Prompt synthesized (ui): {intent[:80]}",
                content=f"Target tool: {self._tool}. Matched {context_len} context node(s).",
                category="methodology",
                tags=["generate", "ui", self._tool],
            )
        except Exception:
            logger.debug("vibe ui could not record memory", exc_info=True)

    # -- copy --------------------------------------------------------------
    def action_copy(self) -> None:
        if not self._generated:
            self.notify("Nothing generated yet.", severity="warning")
            return
        if _copy_to_clipboard(self._generated):
            self.notify("Copied to clipboard.")
        else:
            self.notify(
                "Clipboard unavailable — select the text to copy manually.",
                severity="warning",
            )


def run_ui(settings: VibeSettings, tool: TargetTool) -> int:
    """Launch the Textual app. Returns a process exit code."""
    VibeUIApp(settings, tool).run()
    return 0
