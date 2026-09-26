"""Tests for the read-only ``strix vibe`` prompt orchestrator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from strix.vibe.builder import PromptBuilder
from strix.vibe.config import VibeSettings
from strix.vibe.graph import GraphifyManager
from strix.vibe.llm import NoModelConfiguredError
from strix.vibe.memory import VibeMemory
from strix.vibe.paths import VibeWriteError, assert_write_allowed, safe_write_text
from strix.vibe.synthesizer import PromptSynthesizer


def _fixture_project(root: Path) -> dict[str, str]:
    (root / "app.py").write_text(
        "import os\n\n"
        "class LoginController:\n"
        "    def login(self, user):\n"
        "        return user\n\n"
        "def rate_limit(n):\n"
        "    return n\n",
        encoding="utf-8",
    )
    (root / "utils.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }


def _ast_manager(root: Path) -> GraphifyManager:
    """A manager pinned to the pure-Python ast fallback (no tree-sitter needed)."""
    settings = VibeSettings(project_root=root)
    manager = GraphifyManager(settings)
    manager._load_graphify = lambda: None  # type: ignore[method-assign]
    return manager


# -- write guard -----------------------------------------------------------
def test_assert_write_allowed_permits_inside(tmp_path: Path) -> None:
    out = tmp_path / ".vibe"
    resolved = assert_write_allowed(out / "prompts" / "x.md", [out])
    assert resolved == (out / "prompts" / "x.md").resolve()


def test_assert_write_allowed_rejects_outside(tmp_path: Path) -> None:
    out = tmp_path / ".vibe"
    with pytest.raises(VibeWriteError):
        assert_write_allowed(tmp_path / "app.py", [out])


def test_safe_write_text_atomic(tmp_path: Path) -> None:
    out = tmp_path / ".vibe"
    target = safe_write_text(out / "note.txt", "hello", [out])
    assert target.read_text(encoding="utf-8") == "hello"
    with pytest.raises(VibeWriteError):
        safe_write_text(tmp_path / "escape.txt", "nope", [out])


# -- read-only invariant ---------------------------------------------------
def test_index_and_prompt_never_touch_source(tmp_path: Path) -> None:
    before = _fixture_project(tmp_path)
    manager = _ast_manager(tmp_path)
    manager.ensure_index(force=True)

    context = manager.query("add rate limiting to login")
    PromptBuilder(VibeSettings(project_root=tmp_path)).build(
        intent="add rate limiting to login", tool="claude", context_nodes=context
    )

    after = {
        str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(tmp_path.rglob("*.py"))
    }
    assert before == after, "source files were modified"

    # Every non-source file created lives under .vibe/.
    strays = [
        p.relative_to(tmp_path)
        for p in tmp_path.rglob("*")
        if p.is_file() and ".vibe" not in p.parts and str(p.relative_to(tmp_path)) not in before
    ]
    assert not strays, f"writes escaped .vibe/: {strays}"
    assert (tmp_path / ".vibe" / "graphify-out" / "graph.json").exists()


# -- graph fallback + retrieval -------------------------------------------
def test_ast_fallback_builds_minimal_graph(tmp_path: Path) -> None:
    _fixture_project(tmp_path)
    manager = _ast_manager(tmp_path)
    manager.ensure_index(force=True)
    graph = manager.load_graph()

    assert graph.get("graph", {}).get("builder") == "vibe-ast-fallback"
    labels = {n["label"] for n in graph["nodes"]}
    assert "LoginController" in labels
    assert "rate_limit" in labels


def test_query_ranks_relevant_symbols(tmp_path: Path) -> None:
    _fixture_project(tmp_path)
    manager = _ast_manager(tmp_path)
    manager.ensure_index(force=True)

    hits = manager.query("rate limit the login controller")
    labels = [h["label"] for h in hits]
    assert "rate_limit" in labels
    assert "LoginController" in labels


def test_import_graph_has_no_docker_or_runtime() -> None:
    # Importing the vibe path must not drag in the scan/Docker runtime. Checked in
    # a fresh subprocess: the full test suite imports strix.runtime elsewhere, so
    # an in-process sys.modules check would be polluted.
    code = (
        "import sys; import strix.vibe.cli; "
        "assert 'strix.runtime' not in sys.modules, 'strix.runtime leaked'; "
        "assert 'docker' not in sys.modules, 'docker leaked'"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# -- memory ----------------------------------------------------------------
def test_memory_roundtrip(tmp_path: Path) -> None:
    settings = VibeSettings(project_root=tmp_path)
    memory = VibeMemory(settings)
    created = memory.add("Auth done", "login works", category="progress", tags=["auth"])

    reloaded = VibeMemory(settings)  # fresh instance reads from disk
    fetched = reloaded.get(created["id"])
    assert fetched is not None
    assert fetched["title"] == "Auth done"
    assert reloaded.list(category="progress")[0]["id"] == created["id"]
    assert reloaded.list(search="login")[0]["id"] == created["id"]
    assert reloaded.list(tags=["auth"])[0]["id"] == created["id"]


def test_memory_rejects_bad_input(tmp_path: Path) -> None:
    memory = VibeMemory(VibeSettings(project_root=tmp_path))
    with pytest.raises(ValueError, match="category"):
        memory.add("t", "c", category="nope")
    with pytest.raises(ValueError, match="empty"):
        memory.add("  ", "c")


def test_memory_writes_stay_in_vibe(tmp_path: Path) -> None:
    memory = VibeMemory(VibeSettings(project_root=tmp_path))
    memory.add("t", "c")
    assert (tmp_path / ".vibe" / "memory.json").exists()


# -- builder ---------------------------------------------------------------
@pytest.mark.parametrize("tool", ["claude", "chatgpt", "cursor", "generic"])
def test_build_prompt_has_all_sections(tmp_path: Path, tool: str) -> None:
    builder = PromptBuilder(VibeSettings(project_root=tmp_path))
    context = [
        {"id": "app.py::rate_limit", "label": "rate_limit", "kind": "function",
         "source_file": "app.py", "neighbours": ["app.py"]}
    ]
    brief = builder.build(intent="add rate limiting", tool=tool, context_nodes=context)  # type: ignore[arg-type]
    for heading in ("## Role", "## Objective", "## Context map", "## Constraints",
                    "## Acceptance criteria", "## Output format"):
        assert heading in brief
    assert "rate_limit" in brief
    assert "`app.py`" in brief


def test_build_prompt_handles_no_context(tmp_path: Path) -> None:
    builder = PromptBuilder(VibeSettings(project_root=tmp_path))
    brief = builder.build(intent="do a thing", tool="claude", context_nodes=[])
    assert "No files were matched" in brief


# -- synthesizer / generate ------------------------------------------------
def test_synthesizer_user_message_includes_inputs(tmp_path: Path) -> None:
    synth = PromptSynthesizer(VibeSettings(project_root=tmp_path))
    msg = synth.build_user_message(
        intent="add rate limiting", context_brief="BRIEF-CONTENT-XYZ", tool="claude"
    )
    assert "add rate limiting" in msg
    assert "BRIEF-CONTENT-XYZ" in msg
    assert "claude" in msg


def test_synthesizer_strips_thinking_preamble() -> None:
    from strix.vibe.synthesizer import strip_thinking_preamble

    raw = (
        "**My Approach to the Task**\n"
        "Okay, let me reason about this first...\n\n"
        "# Objective\nImplement the thing.\n"
    )
    cleaned = strip_thinking_preamble(raw)
    assert cleaned.startswith("# Objective")
    assert "My Approach" not in cleaned
    # No heading at all -> returned unchanged (never mangle).
    assert strip_thinking_preamble("just some text") == "just some text"


def test_generate_writes_generated_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    async def fake_completion(system: str, user: str) -> str:
        assert "Context Map" in user  # the brief is passed through
        return "# Objective\nDetailed synthesized prompt.\n"

    monkeypatch.setattr("strix.vibe.synthesizer.run_completion", fake_completion)

    from strix.vibe import cli

    rc = cli.run_vibe(
        ["generate", "add", "rate", "limiting", "--target", str(tmp_path), "--no-index"]
    )
    assert rc == 0
    out = tmp_path / ".vibe" / "prompts" / "add-rate-limiting.generated.md"
    assert out.exists()
    assert "Detailed synthesized prompt" in out.read_text(encoding="utf-8")


def test_generate_hard_fails_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_model(system: str, user: str) -> str:
        raise NoModelConfiguredError

    monkeypatch.setattr("strix.vibe.synthesizer.run_completion", no_model)

    from strix.vibe import cli

    rc = cli.run_vibe(["generate", "x", "--target", str(tmp_path), "--no-index"])
    assert rc == 1


# -- voice -----------------------------------------------------------------
def test_encode_wav_roundtrip() -> None:
    import io
    import wave

    from strix.vibe.voice import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH_BYTES, encode_wav

    pcm = b"\x00\x01" * 1600  # 1600 int16 frames
    wav = encode_wav(pcm)
    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.getnchannels() == CHANNELS
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getsampwidth() == SAMPLE_WIDTH_BYTES
        assert handle.getnframes() == 1600


def test_has_audio_detects_empty() -> None:
    from strix.vibe.voice import encode_wav, has_audio

    assert has_audio(encode_wav(b"\x00\x01" * 100))
    assert not has_audio(encode_wav(b""))
    assert not has_audio(b"not a wav")


async def test_transcribe_builds_audio_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import base64
    import types

    import litellm

    from strix.vibe.voice import encode_wav, transcribe

    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> object:
        captured.update(kwargs)
        message = types.SimpleNamespace(content="  add rate limiting  ")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    settings = VibeSettings(project_root=tmp_path, stt_model="gemini/gemini-2.0-flash")
    wav = encode_wav(b"\x00\x01" * 800)
    text = await transcribe(wav, settings)

    assert text == "add rate limiting"
    assert captured["model"] == "gemini/gemini-2.0-flash"
    parts = captured["messages"][0]["content"]  # type: ignore[index]
    audio = [p for p in parts if p.get("type") == "input_audio"]
    assert len(audio) == 1
    assert audio[0]["input_audio"]["format"] == "wav"
    assert base64.b64decode(audio[0]["input_audio"]["data"]) == wav


async def test_transcribe_skips_empty_audio(tmp_path: Path) -> None:
    from strix.vibe.voice import encode_wav, transcribe

    # No frames -> returns "" without calling any provider.
    assert await transcribe(encode_wav(b""), VibeSettings(project_root=tmp_path)) == ""


def test_recorder_reports_missing_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from strix.vibe.voice import Recorder, VoiceUnavailableError

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sounddevice":
            raise ImportError("no sounddevice")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(VoiceUnavailableError, match="pip install sounddevice"):
        Recorder().start()


def test_voice_command_is_registered() -> None:
    from strix.vibe import cli

    args = cli.build_parser().parse_args(["voice"])
    assert args.func is cli._cmd_voice
    # --voice makes the positional intent optional on generate
    gen = cli.build_parser().parse_args(["generate", "--voice"])
    assert gen.voice is True
    assert gen.intent == []


# -- conversation ----------------------------------------------------------
def test_strip_reasoning_removes_thinking_preamble() -> None:
    from strix.vibe.conversation import strip_reasoning

    # Verbatim shape captured from a live gemini-3-flash-preview reply.
    raw = (
        "**My Thoughts on Token Refreshing**\n\nOkay, so the question is about where "
        "token refreshing happens, right? Looking at this context, the `refresh_tokens()` "
        "function looks like the central piece.\n\n\n\n"
        "Token refreshing is handled by the refresh tokens function in the codex config module."
    )
    cleaned = strip_reasoning(raw)
    assert cleaned.startswith("Token refreshing is handled")
    assert "My Thoughts" not in cleaned
    assert "Okay, so" not in cleaned

    # A normal single-block answer is untouched.
    plain = "It lives in the codex config module."
    assert strip_reasoning(plain) == plain

    # A genuine multi-paragraph answer (no thinking opener) is preserved whole.
    multi = "First point about the code.\n\n\nSecond point about the tests."
    assert strip_reasoning(multi) == multi


def test_format_context_renders_nodes_and_handles_empty() -> None:
    from strix.vibe.conversation import format_context

    assert "no matching code" in format_context([])
    rendered = format_context(
        [{"label": "refresh_tokens()", "kind": "code", "source_file": "strix/config/codex.py"}]
    )
    assert "refresh_tokens()" in rendered
    assert "strix/config/codex.py" in rendered


async def test_chat_session_keeps_history_and_grounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from strix.vibe.conversation import ChatSession

    captured: list[str] = []

    async def fake_completion(system: str, user: str, **kwargs: object) -> str:
        captured.append(user)
        assert kwargs.get("reasoning_effort") == "none"  # spoken replies stay snappy
        return "It lives in the codex config module."

    monkeypatch.setattr("strix.vibe.conversation.run_completion", fake_completion)

    settings = VibeSettings(project_root=tmp_path)
    manager = _ast_manager(tmp_path)
    (tmp_path / "codex.py").write_text("def refresh_tokens():\n    return 1\n", encoding="utf-8")
    manager.ensure_index(force=True)

    session = ChatSession(settings, manager)
    reply = await session.ask("where is refresh tokens handled?")
    assert reply == "It lives in the codex config module."
    # Graph context was injected into the turn.
    assert "Context from the project index" in captured[0]
    assert "refresh_tokens" in captured[0]

    await session.ask("what about the tests?")
    # Second turn carries the prior dialogue so follow-ups resolve.
    assert "Conversation so far" in captured[1]
    assert "where is refresh tokens handled?" in captured[1]
    assert len(session.history) == 4


async def test_chat_session_ignores_blank_turn(tmp_path: Path) -> None:
    from strix.vibe.conversation import ChatSession

    session = ChatSession(VibeSettings(project_root=tmp_path), _ast_manager(tmp_path))
    assert await session.ask("   ") == ""
    assert session.history == []


def test_speak_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    import strix.vibe.voice as voice_mod

    # Empty text never invokes a backend.
    assert voice_mod.speak("   ") is False

    # Every backend failing degrades to False rather than raising.
    def boom(_text: str, _timeout: float) -> bool:
        raise RuntimeError("no engine")

    monkeypatch.setattr(voice_mod, "_speak_windows", boom)
    monkeypatch.setattr(voice_mod, "_speak_unix", boom)
    monkeypatch.setattr(voice_mod, "_speak_pyttsx3", boom)
    assert voice_mod.speak("hello") is False


def test_chat_exit_words() -> None:
    from strix.vibe.cli import _is_exit

    for word in ("quit", "Exit", "stop.", "Goodbye!", "bye"):
        assert _is_exit(word)
    assert not _is_exit("stop the rate limiter from failing")


def test_brief_error_collapses_traceback_and_hints() -> None:
    from strix.vibe.cli import _brief_error

    real = (
        "litellm.APIConnectionError: Missing Gemini API key. Set the GEMINI_API_KEY "
        "or GOOGLE_API_KEY environment variable.\n"
        "Traceback (most recent call last):\n"
        '  File "litellm/main.py", line 656, in acompletion\n'
        "    response = await init_response\n"
        "ValueError: Missing Gemini API key."
    )
    out = _brief_error(Exception(real))
    assert "Traceback" not in out
    assert "Missing Gemini API key" in out
    assert "$env:GEMINI_API_KEY" in out  # actionable hint for the quoting mistake

    # Unrelated failures stay short and hint-free.
    plain = _brief_error(RuntimeError("connection reset"))
    assert plain == "connection reset"

    # Rate limits get a wait-and-retry hint (free-tier's most common failure).
    rl = _brief_error(RuntimeError("litellm.RateLimitError: geminiException 429"))
    assert "rate limit" in rl.lower()
    assert "wait" in rl.lower()


def test_chat_command_is_registered() -> None:
    from strix.vibe import cli

    args = cli.build_parser().parse_args(["chat", "--text", "--no-speak"])
    assert args.func is cli._cmd_chat
    assert args.text is True
    assert args.no_speak is True
    assert args.allow_self is False


# -- self-repo guard & planning phase ---------------------------------------
def test_is_tool_own_repo_detects_this_repo() -> None:
    import strix
    from strix.vibe.cli import _is_tool_own_repo

    repo_root = Path(strix.__file__).resolve().parent.parent
    assert _is_tool_own_repo(repo_root) is True


def test_is_tool_own_repo_false_for_unrelated_dir(tmp_path: Path) -> None:
    from strix.vibe.cli import _is_tool_own_repo

    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    assert _is_tool_own_repo(tmp_path) is False


def test_has_source_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty-project"
    nonexistent = tmp_path / "does-not-exist"
    populated = tmp_path / "real-project"
    populated.mkdir()
    (populated / "app.py").write_text("x = 1\n", encoding="utf-8")

    assert GraphifyManager(VibeSettings(project_root=nonexistent)).has_source_files() is False
    empty.mkdir()
    assert GraphifyManager(VibeSettings(project_root=empty)).has_source_files() is False
    assert GraphifyManager(VibeSettings(project_root=populated)).has_source_files() is True


def test_chat_refuses_own_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    import strix
    from strix.vibe import cli

    called = False

    def fail_if_called(*_a: object, **_k: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("should not reach session selection")

    monkeypatch.setattr(cli, "_select_chat_session", fail_if_called)

    repo_root = Path(strix.__file__).resolve().parent.parent
    rc = cli.run_vibe(["chat", "--target", str(repo_root)])
    assert rc == 1
    assert called is False


def test_chat_allow_self_bypasses_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import strix
    from strix.vibe import cli

    class ImmediatelyDoneError(Exception):
        pass

    def fake_select(settings: object, *, no_index: bool) -> None:
        raise ImmediatelyDoneError

    monkeypatch.setattr(cli, "_select_chat_session", fake_select)
    repo_root = Path(strix.__file__).resolve().parent.parent
    with pytest.raises(ImmediatelyDoneError):
        cli.run_vibe(["chat", "--target", str(repo_root), "--allow-self"])


async def test_planning_session_asks_and_remembers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from strix.vibe.memory import VibeMemory
    from strix.vibe.planning import PlanningSession

    async def fake_completion(system: str, user: str, **kwargs: object) -> str:
        assert "no code yet" in system.lower() or "new" in system.lower()
        return "What should this project do for its users?"

    monkeypatch.setattr("strix.vibe.planning.run_completion", fake_completion)

    settings = VibeSettings(project_root=tmp_path / "brand-new-idea")
    session = PlanningSession(settings)
    reply = await session.ask("I want to build a recipe organizer app")
    assert "users" in reply.lower()
    assert len(session.history) == 2

    notes = VibeMemory(settings).list(category="plan")
    assert len(notes) == 1
    assert "recipe organizer" in notes[0]["content"]


async def test_planning_session_ignores_blank_turn(tmp_path: Path) -> None:
    from strix.vibe.planning import PlanningSession

    session = PlanningSession(VibeSettings(project_root=tmp_path))
    assert await session.ask("   ") == ""
    assert session.history == []


async def test_planning_session_generate_produces_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from strix.vibe.planning import PlanningSession

    async def fake_completion(system: str, user: str) -> str:
        assert "recipe organizer" in user.lower()
        return (
            "## Concept & Goals\nA recipe organizer.\n\n"
            "## Proposed Features\nSave recipes.\n\n"
            "## Suggested Tech Stack\nto be decided\n\n"
            "## Implementation Milestones\nStep 1.\n\n"
            "## Open Questions\nWhich platform?"
        )

    monkeypatch.setattr("strix.vibe.planning.run_completion", fake_completion)

    settings = VibeSettings(project_root=tmp_path / "brand-new-idea")
    session = PlanningSession(settings)
    session._history = [
        {"role": "user", "content": "I want a recipe organizer"},
        {"role": "assistant", "content": "Who is it for?"},
    ]
    plan = await session.generate("claude")
    assert "Concept & Goals" in plan
    assert "Open Questions" in plan


async def test_planning_session_generate_requires_discussion(tmp_path: Path) -> None:
    from strix.vibe.planning import PlanningSession

    session = PlanningSession(VibeSettings(project_root=tmp_path))
    with pytest.raises(RuntimeError, match="nothing has been discussed"):
        await session.generate("claude")


def test_generate_trigger_detection() -> None:
    from strix.vibe.cli import _is_generate_trigger

    assert _is_generate_trigger("generate a prompt for that")
    assert _is_generate_trigger("let's turn this into a prompt")
    assert _is_generate_trigger("write up the plan")
    assert not _is_generate_trigger("what's the plan for testing")
    assert not _is_generate_trigger("where is the login handled")


async def test_chat_session_generate_reuses_synthesis_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from strix.vibe.conversation import ChatSession

    async def fake_synthesize(self: object, *, intent: str, context_brief: str, tool: str) -> str:
        assert "rate_limit" in intent or "login" in intent.lower() or intent  # non-empty
        return "# Objective\nDo the thing.\n"

    monkeypatch.setattr(
        "strix.vibe.synthesizer.PromptSynthesizer.synthesize", fake_synthesize
    )

    settings = VibeSettings(project_root=tmp_path)
    _fixture_project(tmp_path)
    manager = _ast_manager(tmp_path)
    manager.ensure_index(force=True)
    session = ChatSession(settings, manager)
    session._history = [
        {"role": "user", "content": "where is rate limiting handled"},
        {"role": "assistant", "content": "In rate_limit()."},
    ]
    result = await session.generate("claude")
    assert result.startswith("# Objective")


async def test_chat_session_generate_requires_discussion(tmp_path: Path) -> None:
    from strix.vibe.conversation import ChatSession

    session = ChatSession(VibeSettings(project_root=tmp_path), _ast_manager(tmp_path))
    with pytest.raises(RuntimeError, match="nothing has been discussed"):
        await session.generate("claude")


def test_cmd_chat_planning_mode_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """New-project target -> planning banner, not analyze-mode banner."""
    from strix.vibe import cli

    turns = iter(["I want to build a recipe organizer", "quit"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(turns))

    async def fake_completion(system: str, user: str, **kwargs: object) -> str:
        return "Sounds good, who is it for?"

    monkeypatch.setattr("strix.vibe.planning.run_completion", fake_completion)

    rc = cli.run_vibe(
        [
            "chat",
            "--target",
            str(tmp_path / "brand-new-idea"),
            "--text",
            "--no-speak",
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "switching to project-planning mode" in err
    assert "Analyzing existing project" not in err


# -- interactive UI --------------------------------------------------------
def test_ui_command_is_registered() -> None:
    from strix.vibe import cli

    args = cli.build_parser().parse_args(["ui", "--tool", "claude"])
    assert args.func is cli._cmd_ui


async def test_ui_search_and_generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("textual")
    from textual.widgets import DataTable, Input, TextArea

    from strix.vibe.ui import _SEARCH_DEBOUNCE_S, VibeUIApp

    settings = VibeSettings(project_root=tmp_path)
    # Seed a graph so the on-mount indexing is an instant reuse (no graphify).
    graph = {
        "directed": False,
        "graph": {"builder": "seed"},
        "nodes": [
            {
                "id": "auth.py::rate_limit",
                "label": "rate_limit",
                "kind": "function",
                "source_file": "auth.py",
            }
        ],
        "edges": [],
    }
    settings.graph_out_dir.mkdir(parents=True, exist_ok=True)
    safe_write_text(settings.graph_json_path, json.dumps(graph), settings.allowed_write_roots)

    async def fake_completion(system: str, user: str) -> str:
        return "# Objective\nUI-GENERATED-PROMPT\n"

    monkeypatch.setattr("strix.vibe.synthesizer.run_completion", fake_completion)

    app = VibeUIApp(settings, "claude")
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()  # indexing

        app.query_one("#intent", Input).value = "rate limit"
        await pilot.pause(_SEARCH_DEBOUNCE_S + 0.15)
        await app.workers.wait_for_complete()  # live search
        assert app.query_one("#matches", DataTable).row_count >= 1

        await pilot.press("ctrl+g")
        await app.workers.wait_for_complete()  # synthesis
        assert "UI-GENERATED-PROMPT" in app.query_one("#output", TextArea).text

    # The generated prompt was saved under .vibe/prompts.
    assert (settings.prompts_dir / "rate-limit.generated.md").exists()


async def test_ui_f6_records_and_fills_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("textual")
    from textual.widgets import Input

    import strix.vibe.voice as voice_mod
    from strix.vibe.ui import VibeUIApp

    class FakeRecorder:
        def __init__(self) -> None:
            self.is_recording = False

        def start(self) -> None:
            self.is_recording = True

        def stop(self) -> bytes:
            self.is_recording = False
            return voice_mod.encode_wav(b"\x00\x01" * 400)

    async def fake_transcribe(wav: bytes, settings: object) -> str:
        return "spoken intent text"

    monkeypatch.setattr(voice_mod, "Recorder", FakeRecorder)
    monkeypatch.setattr(voice_mod, "transcribe", fake_transcribe)

    settings = VibeSettings(project_root=tmp_path)
    settings.graph_out_dir.mkdir(parents=True, exist_ok=True)
    safe_write_text(
        settings.graph_json_path,
        json.dumps({"nodes": [], "edges": [], "graph": {}}),
        settings.allowed_write_roots,
    )

    app = VibeUIApp(settings, "claude")
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.press("f6")  # start recording
        assert app._recorder is not None
        await pilot.press("f6")  # stop -> transcribe
        await app.workers.wait_for_complete()
        assert app.query_one("#intent", Input).value == "spoken intent text"


def test_generate_from_brief_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    brief_file = tmp_path / "brief.md"
    brief_file.write_text("# Context map\n- `app.py`\n", encoding="utf-8")

    captured: dict[str, str] = {}

    async def fake_completion(system: str, user: str) -> str:
        captured["user"] = user
        return "# Objective\nfrom brief\n"

    monkeypatch.setattr("strix.vibe.synthesizer.run_completion", fake_completion)

    from strix.vibe import cli

    rc = cli.run_vibe(
        ["generate", "do", "it", "--target", str(tmp_path), "--from-brief", str(brief_file)]
    )
    assert rc == 0
    assert "app.py" in captured["user"]  # the supplied brief was used verbatim
