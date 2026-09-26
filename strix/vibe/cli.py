"""``strix vibe`` — read-only project analysis + external-prompt generation.

Subcommands:

- ``index`` — build/refresh the read-only knowledge graph under ``.vibe/``.
- ``prompt`` — turn an intent + project context into a copy-pasteable prompt.
- ``memory`` — inspect/append the project memory (``list`` / ``show`` / ``add``).

Nothing here touches the user's source tree; all writes go under ``.vibe/``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from strix.vibe.builder import PromptBuilder
from strix.vibe.config import VibeSettings
from strix.vibe.graph import GraphifyManager
from strix.vibe.llm import NoModelConfiguredError
from strix.vibe.memory import VALID_CATEGORIES, VibeMemory
from strix.vibe.paths import safe_write_text
from strix.vibe.synthesizer import PromptSynthesizer


if TYPE_CHECKING:
    from strix.vibe.conversation import ChatSession
    from strix.vibe.planning import PlanningSession


logger = logging.getLogger(__name__)

_MAX_MEMORY_NOTES_IN_PROMPT = 8
_SLUG_MAX = 48


def _settings(target: str | None) -> VibeSettings:
    if target:
        return VibeSettings(project_root=Path(target))
    return VibeSettings()


def _is_tool_own_repo(project_root: Path) -> bool:
    """Whether *project_root* is (or looks like) vibe's own installation repo.

    ``chat`` defaults to cwd when no ``--target`` is given; the classic accident
    is running it from inside this repo, where it would describe itself instead
    of acting as a prompt-authoring partner for the user's project. Checked two
    ways so a renamed clone is still caught: exact match against the installed
    ``strix`` package's repo root, and presence of the vendored engine.
    """
    import strix

    try:
        repo_root = Path(strix.__file__).resolve().parent.parent
        resolved = project_root.resolve()
    except OSError:
        return False
    if resolved == repo_root:
        return True
    return (resolved / "strix" / "vibe" / "_vendor" / "graphify").exists()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug[:_SLUG_MAX].rstrip("-")) or "prompt"


def _copy_to_clipboard(text: str) -> bool:
    """Best-effort clipboard copy. Never raises (headless/SSH safe)."""
    try:
        import pyperclip
    except ImportError:
        return False
    try:
        pyperclip.copy(text)
    except Exception:  # noqa: BLE001 - no display / no clipboard backend, etc.
        return False
    return True


def _cmd_index(args: argparse.Namespace) -> int:
    settings = _settings(args.path)
    manager = GraphifyManager(settings)
    path = manager.ensure_index(force=args.force)
    graph = manager.load_graph()
    print(
        f"Indexed {settings.project_root} -> {path} "
        f"({len(graph.get('nodes', []))} nodes)."
    )
    return 0


def _build_brief(
    settings: VibeSettings, intent: str, tool: str, *, no_index: bool
) -> tuple[str, list[dict[str, object]], VibeMemory]:
    """Shared context pipeline: (re)index -> query -> memory -> deterministic brief."""
    manager = GraphifyManager(settings)
    if not no_index:
        manager.ensure_index(force=False)
    context = manager.query(intent)
    memory = VibeMemory(settings)
    notes = memory.list()[:_MAX_MEMORY_NOTES_IN_PROMPT]
    brief = PromptBuilder(settings).build(
        intent=intent, tool=tool, context_nodes=context, memory_notes=notes
    )
    return brief, context, memory


def _emit(
    settings: VibeSettings,
    *,
    intent: str,
    tool: str,
    text: str,
    out: str | None,
    default_suffix: str,
    context_len: int,
    record_title: str,
    tags: list[str],
    memory: VibeMemory,
) -> None:
    """Print to stdout, save a copy under .vibe/prompts, clipboard, record in memory."""
    print(text)
    out_path = Path(out) if out else settings.prompts_dir / f"{_slugify(intent)}{default_suffix}"
    try:
        safe_write_text(out_path, text, settings.allowed_write_roots)
        saved: str | None = str(out_path)
    except Exception as exc:  # noqa: BLE001 - writing the copy is a convenience.
        logger.warning("could not save prompt to %s: %s", out_path, exc)
        saved = None

    copied = _copy_to_clipboard(text)

    try:
        memory.add(
            title=record_title,
            content=f"Target tool: {tool}. Matched {context_len} context node(s).",
            category="methodology",
            tags=tags,
        )
    except Exception:
        logger.debug("could not record prompt generation in memory", exc_info=True)

    footer = []
    if saved:
        footer.append(f"saved to {saved}")
    if copied:
        footer.append("copied to clipboard")
    if footer:
        print(f"\n---\n({'; '.join(footer)})", file=sys.stderr)


def _cmd_prompt(args: argparse.Namespace) -> int:
    settings = _settings(args.target)
    intent = " ".join(args.intent).strip()
    if not intent:
        print('error: an intent is required, e.g. strix vibe prompt "add rate limiting"',
              file=sys.stderr)
        return 2
    tool = args.tool or settings.default_tool

    draft, context, memory = _build_brief(settings, intent, tool, no_index=args.no_index)
    if args.polish:
        draft = asyncio.run(PromptBuilder(settings).polish(draft))

    _emit(
        settings,
        intent=intent,
        tool=tool,
        text=draft,
        out=args.out,
        default_suffix=".md",
        context_len=len(context),
        record_title=f"Prompt generated: {intent[:80]}",
        tags=["prompt", tool],
        memory=memory,
    )
    return 0


def _voice_intent(settings: VibeSettings) -> str | None:
    """Record + transcribe an intent. Returns None if unusable (already reported)."""
    from strix.vibe.voice import VoiceUnavailableError

    try:
        text = _record_and_transcribe(settings)
    except VoiceUnavailableError:
        return None
    except Exception as exc:  # noqa: BLE001 - report provider/mic failures cleanly.
        print(f"error: transcription failed: {exc}", file=sys.stderr)
        return None
    if not text:
        print("error: nothing transcribed - was the microphone silent?", file=sys.stderr)
        return None
    print(f"Heard: {text}", file=sys.stderr)
    return text


def _cmd_generate(args: argparse.Namespace) -> int:
    settings = _settings(args.target)
    intent = " ".join(args.intent).strip()
    if getattr(args, "voice", False):
        spoken = _voice_intent(settings)
        if spoken is None:
            return 1
        intent = spoken
    if not intent:
        print('error: an intent is required, e.g. strix vibe generate "add rate limiting"',
              file=sys.stderr)
        return 2
    tool = args.tool or settings.default_tool

    if args.from_brief:
        brief_path = Path(args.from_brief)
        try:
            brief = brief_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: could not read --from-brief {brief_path}: {exc}", file=sys.stderr)
            return 2
        context_len = 0
        memory = VibeMemory(settings)
    else:
        brief, context, memory = _build_brief(settings, intent, tool, no_index=args.no_index)
        context_len = len(context)

    synthesizer = PromptSynthesizer(settings)
    try:
        detailed = asyncio.run(
            synthesizer.synthesize(intent=intent, context_brief=brief, tool=tool)
        )
    except NoModelConfiguredError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report provider failures cleanly.
        print(f"error: prompt synthesis failed: {exc}", file=sys.stderr)
        return 1

    _emit(
        settings,
        intent=intent,
        tool=tool,
        text=detailed,
        out=args.out,
        default_suffix=".generated.md",
        context_len=context_len,
        record_title=f"Prompt synthesized: {intent[:80]}",
        tags=["generate", tool],
        memory=memory,
    )
    return 0


def _cmd_memory_list(args: argparse.Namespace) -> int:
    memory = VibeMemory(_settings(args.target))
    notes = memory.list(category=args.category, tags=args.tag, search=args.search)
    if not notes:
        print("(no memory entries)")
        return 0
    for note in notes:
        tags = f" [{', '.join(note['tags'])}]" if note.get("tags") else ""
        print(f"{note['id']}  {note['category']:<12}  {note['title']}{tags}")
    return 0


def _cmd_memory_show(args: argparse.Namespace) -> int:
    memory = VibeMemory(_settings(args.target))
    note = memory.get(args.id)
    if note is None:
        print(f"error: no memory entry with id {args.id!r}", file=sys.stderr)
        return 1
    print(f"# {note['title']}  ({note['category']})")
    if note.get("tags"):
        print(f"tags: {', '.join(note['tags'])}")
    print(f"created: {note['created_at']}")
    print()
    print(note["content"])
    return 0


def _cmd_memory_add(args: argparse.Namespace) -> int:
    memory = VibeMemory(_settings(args.target))
    try:
        note = memory.add(
            title=args.title,
            content=args.content,
            category=args.category,
            tags=args.tag,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"added {note['id']} ({note['category']}): {note['title']}")
    return 0


def _record_and_transcribe(settings: VibeSettings) -> str:
    """Record until the user presses Enter, then transcribe. Returns the text."""
    from strix.vibe.voice import Recorder, VoiceUnavailableError, transcribe

    recorder = Recorder()
    try:
        recorder.start()
    except VoiceUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
    print("Recording... speak now, then press Enter to stop.", file=sys.stderr)
    with contextlib.suppress(EOFError, KeyboardInterrupt):
        input()
    wav = recorder.stop()
    print("Transcribing...", file=sys.stderr)
    return asyncio.run(transcribe(wav, settings))


def _cmd_voice(args: argparse.Namespace) -> int:
    from strix.vibe.voice import VoiceUnavailableError

    settings = _settings(args.target)
    try:
        text = _record_and_transcribe(settings)
    except VoiceUnavailableError:
        return 1
    except Exception as exc:  # noqa: BLE001 - report provider/mic failures cleanly.
        print(f"error: transcription failed: {exc}", file=sys.stderr)
        return 1
    if not text:
        print("(nothing transcribed - was the microphone silent?)", file=sys.stderr)
        return 1
    print(text)
    if getattr(args, "speak", False):
        from strix.vibe.voice import speak

        speak(text)
    return 0


_EXIT_WORDS = frozenset({"quit", "exit", "stop", "goodbye", "bye", "that's all", "thats all"})
_ERROR_SUMMARY_CHARS = 200


def _is_exit(utterance: str) -> bool:
    cleaned = utterance.strip().lower().rstrip(".!?")
    return cleaned in _EXIT_WORDS


def _brief_error(exc: Exception) -> str:
    """One readable line from a provider exception.

    LiteLLM stringifies a full traceback into its error message; printing that
    mid-conversation buries the actual cause. Keep the first meaningful line and
    append a hint when it is the usual missing-credentials case.
    """
    message = str(exc).strip()
    first = next(
        (
            line.strip()
            for line in message.splitlines()
            if line.strip() and not line.startswith(("Traceback", "  ", "\t"))
        ),
        message,
    )
    first = first[:_ERROR_SUMMARY_CHARS]
    lowered = message.lower()
    if "api key" in lowered:
        first += (
            '\n  hint: set it with quotes -> $env:GEMINI_API_KEY = "your-key"  '
            "(then check $env:GEMINI_API_KEY.Length)"
        )
    elif "ratelimit" in lowered or "rate limit" in lowered or "429" in lowered:
        first = (
            "Gemini free-tier rate limit reached (too many requests too fast).\n"
            "  hint: wait ~30-60s and try again, or type instead of speaking to "
            "make fewer calls."
        )
    return first


_GENERATE_VERB_RE = re.compile(r"\b(generate|write up|turn\b.*\binto)\b", re.IGNORECASE)
_GENERATE_NOUN_RE = re.compile(r"\b(prompt|plan)\b", re.IGNORECASE)


def _is_generate_trigger(utterance: str) -> bool:
    return bool(_GENERATE_VERB_RE.search(utterance) and _GENERATE_NOUN_RE.search(utterance))


def _select_chat_session(
    settings: VibeSettings, *, no_index: bool
) -> ChatSession | PlanningSession:
    """Pick analyze mode (existing project) or planning mode (nothing to index yet),
    printing a one-line banner so the active mode is never ambiguous."""
    from strix.vibe.conversation import ChatSession
    from strix.vibe.planning import PlanningSession

    manager = GraphifyManager(settings)
    if not manager.has_source_files():
        print(
            f"No existing project found at {settings.project_root} - "
            "switching to project-planning mode.",
            file=sys.stderr,
        )
        return PlanningSession(settings)

    if not no_index:
        print("Indexing project (first run can take a while)...", file=sys.stderr)
        manager.ensure_index(force=False)
    print(f"Analyzing existing project at {settings.project_root}.", file=sys.stderr)
    return ChatSession(settings, manager)


def _cmd_chat(args: argparse.Namespace) -> int:
    """Hold a spoken back-and-forth conversation about the project."""
    from strix.vibe.voice import VoiceUnavailableError, speak

    settings = _settings(args.target)
    if not args.allow_self and _is_tool_own_repo(settings.project_root):
        print(
            "error: this looks like vibe's own repository - point --target at the "
            "project you want to work on (or pass --allow-self to discuss vibe itself).",
            file=sys.stderr,
        )
        return 1

    session = _select_chat_session(settings, no_index=args.no_index)

    typed = args.text
    say_it = not args.no_speak
    print(
        "Chat ready. "
        + (
            "Type your question and press Enter"
            if typed
            else "Type a question, OR press Enter on an empty line to speak it"
        )
        + "; 'quit' to leave.",
        file=sys.stderr,
    )

    while True:
        try:
            utterance = _next_utterance(settings, typed=typed)
        except VoiceUnavailableError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except (EOFError, KeyboardInterrupt):
            print("\nBye.", file=sys.stderr)
            return 0
        except Exception as exc:  # noqa: BLE001 - a bad turn shouldn't end the chat.
            print(f"(could not capture that: {_brief_error(exc)})", file=sys.stderr)
            continue

        if not utterance:
            print("(nothing heard - try again)", file=sys.stderr)
            continue
        print(f"\nYou: {utterance}")
        if _is_exit(utterance):
            print("Assistant: Goodbye.")
            if say_it:
                speak("Goodbye.")
            return 0

        if _is_generate_trigger(utterance):
            tool = args.tool or settings.default_tool
            outcome = _generate_from_chat(settings, session, utterance, tool, say_it=say_it)
            if outcome is not None:
                return outcome
            continue

        outcome = _answer_turn(session, utterance, say_it=say_it)
        if outcome is not None:
            return outcome


def _generate_from_chat(
    settings: VibeSettings,
    session: ChatSession | PlanningSession,
    utterance: str,
    tool: str,
    *,
    say_it: bool,
) -> int | None:
    """Synthesize an artifact from the conversation. Returns an exit code to
    stop, or None to keep chatting (mirrors ``_answer_turn``'s contract)."""
    from strix.vibe.planning import PlanningSession
    from strix.vibe.voice import speak

    print("Generating...", file=sys.stderr)
    try:
        artifact = asyncio.run(session.generate(tool))  # type: ignore[arg-type]
    except NoModelConfiguredError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report and keep the chat alive.
        print(f"(sorry, generation failed: {_brief_error(exc)})", file=sys.stderr)
        return None

    is_planning = isinstance(session, PlanningSession)
    _emit(
        settings,
        intent=utterance,
        tool=tool,
        text=artifact,
        out=None,
        default_suffix=".plan.generated.md" if is_planning else ".generated.md",
        context_len=0,
        record_title=f"{'Plan' if is_planning else 'Prompt'} synthesized (chat): {utterance[:80]}",
        tags=["chat", "plan" if is_planning else "generate", tool],
        memory=VibeMemory(settings),
    )
    if say_it:
        speak("Generated and saved.")
    return None


def _answer_turn(
    session: ChatSession | PlanningSession, utterance: str, *, say_it: bool
) -> int | None:
    """Answer one turn. Returns an exit code to stop, or None to keep chatting."""
    from strix.vibe.voice import speak

    try:
        reply = asyncio.run(session.ask(utterance))
    except NoModelConfiguredError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - report and keep the chat alive.
        print(f"(sorry, that failed: {_brief_error(exc)})", file=sys.stderr)
        return None
    print(f"Assistant: {reply}\n")
    if say_it and reply:
        speak(reply)
    return None


def _next_utterance(settings: VibeSettings, *, typed: bool) -> str:
    """Get the next user turn. Type text and press Enter to send it; press Enter
    on an empty line to record from the microphone instead (unless --text)."""
    prompt = "You: " if typed else "Type, or press Enter to talk > "
    line = input(prompt).strip()
    if typed or line:
        return line
    return _record_and_transcribe(settings)


def _cmd_ui(args: argparse.Namespace) -> int:
    settings = _settings(args.target)
    tool = args.tool or settings.default_tool
    try:
        from strix.vibe.ui import run_ui
    except ImportError as exc:
        print(
            "error: the interactive UI needs the 'textual' package.\n"
            "  install it with:  pip install textual   (or: pip install .[vibe])",
            file=sys.stderr,
        )
        logger.debug("textual import failed: %s", exc)
        return 1
    return run_ui(settings, tool)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strix vibe",
        description="Read-only project analysis and external-prompt generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build/refresh the read-only project graph.")
    p_index.add_argument("path", nargs="?", default=None, help="Project root (default: cwd).")
    p_index.add_argument("--force", action="store_true", help="Rebuild even if a graph exists.")
    p_index.set_defaults(func=_cmd_index)

    p_prompt = sub.add_parser("prompt", help="Generate a prompt for an external AI tool.")
    p_prompt.add_argument("intent", nargs="+", help="What you want the external tool to do.")
    p_prompt.add_argument(
        "--tool",
        choices=("claude", "chatgpt", "cursor", "generic"),
        default=None,
        help="Target tool (default: VIBE_DEFAULT_TOOL or 'claude').",
    )
    p_prompt.add_argument("--target", default=None, help="Project root (default: cwd).")
    p_prompt.add_argument("--polish", action="store_true", help="Refine via the configured LLM.")
    p_prompt.add_argument(
        "--no-index", action="store_true", help="Skip (re)indexing; use the existing graph."
    )
    p_prompt.add_argument(
        "--out", default=None, help="Write the prompt here (default: .vibe/prompts/)."
    )
    p_prompt.set_defaults(func=_cmd_prompt)

    p_gen = sub.add_parser(
        "generate",
        help="Synthesize a detailed, actionable engineering prompt via the LLM.",
    )
    p_gen.add_argument(
        "intent",
        nargs="*",
        default=[],
        help="The feature/change you want implemented (omit when using --voice).",
    )
    p_gen.add_argument(
        "--voice", action="store_true", help="Speak the intent instead of typing it."
    )
    p_gen.add_argument(
        "--tool",
        choices=("claude", "chatgpt", "cursor", "generic"),
        default=None,
        help="Target tool (default: VIBE_DEFAULT_TOOL or 'claude').",
    )
    p_gen.add_argument("--target", default=None, help="Project root (default: cwd).")
    p_gen.add_argument(
        "--no-index", action="store_true", help="Skip (re)indexing; use the existing graph."
    )
    p_gen.add_argument(
        "--from-brief",
        default=None,
        help="Synthesize from an existing context brief file instead of rebuilding it.",
    )
    p_gen.add_argument(
        "--out", default=None, help="Write the prompt here (default: .vibe/prompts/*.generated.md)."
    )
    p_gen.set_defaults(func=_cmd_generate)

    p_ui = sub.add_parser("ui", help="Interactive terminal UI (needs the 'textual' extra).")
    p_ui.add_argument("--target", default=None, help="Project root (default: cwd).")
    p_ui.add_argument(
        "--tool",
        choices=("claude", "chatgpt", "cursor", "generic"),
        default=None,
        help="Target tool (default: VIBE_DEFAULT_TOOL or 'claude').",
    )
    p_ui.set_defaults(func=_cmd_ui)

    p_voice = sub.add_parser(
        "voice", help="Record from the mic and print the transcript (mic check)."
    )
    p_voice.add_argument("--target", default=None, help="Project root (default: cwd).")
    p_voice.add_argument(
        "--speak", action="store_true", help="Read the transcript back aloud (TTS check)."
    )
    p_voice.set_defaults(func=_cmd_voice)

    p_chat = sub.add_parser(
        "chat", help="Hold a spoken conversation about the project (talks back)."
    )
    p_chat.add_argument("--target", default=None, help="Project root (default: cwd).")
    p_chat.add_argument(
        "--tool",
        choices=("claude", "chatgpt", "cursor", "generic"),
        default=None,
        help="Target tool for anything generated (default: VIBE_DEFAULT_TOOL or 'claude').",
    )
    p_chat.add_argument(
        "--text", action="store_true", help="Type instead of speaking (replies still spoken)."
    )
    p_chat.add_argument("--no-speak", action="store_true", help="Print replies without speaking.")
    p_chat.add_argument(
        "--no-index", action="store_true", help="Skip (re)indexing; use the existing graph."
    )
    p_chat.add_argument(
        "--allow-self",
        action="store_true",
        help="Allow discussing vibe's own repository (normally refused).",
    )
    p_chat.set_defaults(func=_cmd_chat)

    _add_memory_parser(sub)
    return parser


def _add_memory_parser(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p_mem = sub.add_parser("memory", help="Inspect or append project memory.")
    mem_sub = p_mem.add_subparsers(dest="mem_command", required=True)

    m_list = mem_sub.add_parser("list", help="List memory entries.")
    m_list.add_argument("--target", default=None)
    m_list.add_argument("--category", default=None, choices=VALID_CATEGORIES)
    m_list.add_argument("--tag", action="append", default=None)
    m_list.add_argument("--search", default=None)
    m_list.set_defaults(func=_cmd_memory_list)

    m_show = mem_sub.add_parser("show", help="Show one memory entry.")
    m_show.add_argument("id")
    m_show.add_argument("--target", default=None)
    m_show.set_defaults(func=_cmd_memory_show)

    m_add = mem_sub.add_parser("add", help="Add a memory entry.")
    m_add.add_argument("title")
    m_add.add_argument("--content", required=True)
    m_add.add_argument("--category", default="general", choices=VALID_CATEGORIES)
    m_add.add_argument("--tag", action="append", default=None)
    m_add.add_argument("--target", default=None)
    m_add.set_defaults(func=_cmd_memory_add)


def run_vibe(argv: list[str]) -> int:
    """Entry point for ``strix vibe …``. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.exit(run_vibe(sys.argv[1:]))


if __name__ == "__main__":
    main()
