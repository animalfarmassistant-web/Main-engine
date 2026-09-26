"""Voice input for the vibe orchestrator: record from the mic, transcribe to text.

Recording uses ``sounddevice`` (PortAudio) and encodes WAV with the stdlib
``wave`` module -- no extra audio dependency. Transcription posts the audio to a
speech-capable model as an OpenAI-style ``input_audio`` content part, which
LiteLLM converts to Gemini ``inline_data``.

This calls ``litellm.acompletion`` directly rather than going through
``strix.vibe.llm.run_completion``: the agents-SDK ``Model`` interface that
powers the latter is text/tool oriented and has no audio content type.

The transcription model is configured separately from ``STRIX_LLM`` (see
``VibeSettings.stt_model`` / ``VIBE_STT_MODEL``) because not every chat model
accepts audio, and LiteLLM's ``supports_audio_input`` map is unreliable: it
still flags the retired ``gemini-2.0-flash`` while leaving the working
``gemini-3-flash-preview`` unset. Trust live behaviour over the map when
changing the default.
"""

from __future__ import annotations

import base64
import io
import logging
import shutil
import subprocess
import sys
import wave
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from strix.vibe.config import VibeSettings


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # int16

_TRANSCRIBE_INSTRUCTION = (
    "Transcribe the spoken audio verbatim. Return ONLY the transcript text - "
    "no preamble, no quotes, no commentary. If the audio is silent or "
    "unintelligible, return an empty response."
)


class VoiceUnavailableError(RuntimeError):
    """Raised when audio capture is not usable on this machine."""


def _import_sounddevice() -> Any:
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        # OSError: the wheel is present but PortAudio itself failed to load.
        raise VoiceUnavailableError(
            "Microphone capture needs the 'sounddevice' package.\n"
            "  install it with:  pip install sounddevice   (or: pip install .[vibe])"
        ) from exc
    return sounddevice


def encode_wav(frames: bytes) -> bytes:
    """Wrap raw little-endian int16 PCM in a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH_BYTES)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(frames)
    return buffer.getvalue()


class Recorder:
    """Non-blocking mic recorder. ``start()`` then ``stop()`` -> WAV bytes."""

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._stream: Any = None

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        sounddevice = _import_sounddevice()
        self._chunks = []

        def callback(indata: Any, _frames: int, _time: Any, status: Any) -> None:
            if status:
                logger.debug("audio input status: %s", status)
            self._chunks.append(bytes(indata))

        try:
            stream = sounddevice.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                callback=callback,
            )
            stream.start()
        except Exception as exc:
            raise VoiceUnavailableError(
                f"Could not open the microphone: {exc}\n"
                "  check that an input device exists and the terminal has mic permission."
            ) from exc
        self._stream = stream

    def stop(self) -> bytes:
        """Stop recording and return the captured audio as WAV bytes."""
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.debug("error closing audio stream", exc_info=True)
        frames = b"".join(self._chunks)
        self._chunks = []
        return encode_wav(frames)

    def duration_seconds(self) -> float:
        captured = sum(len(chunk) for chunk in self._chunks)
        return captured / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH_BYTES)


def has_audio(wav_bytes: bytes) -> bool:
    """Whether the WAV holds any actual frames (guards empty/instant recordings)."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
            return handle.getnframes() > 0
    except (wave.Error, EOFError):
        return False


async def transcribe(wav_bytes: bytes, settings: VibeSettings) -> str:
    """Transcribe WAV audio to text. Returns "" when nothing was said."""
    if not has_audio(wav_bytes):
        return ""

    import litellm

    from strix.config import load_settings

    llm = load_settings().llm
    encoded = base64.b64encode(wav_bytes).decode("ascii")
    response = await litellm.acompletion(
        model=settings.stt_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _TRANSCRIBE_INSTRUCTION},
                    {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
                ],
            }
        ],
        api_key=llm.api_key or None,
        timeout=llm.timeout,
    )
    return _first_message_text(response).strip()


def speak(text: str, *, timeout: float = 120.0) -> bool:
    """Say *text* aloud. Returns True if speech was produced.

    Best-effort by design: a machine with no speech engine should degrade to
    silent text output, never crash the conversation. Windows uses the built-in
    SAPI engine (no dependency); macOS/Linux fall back to ``say``/``espeak``,
    then to ``pyttsx3`` if installed.
    """
    spoken = text.strip()
    if not spoken:
        return False
    for backend in (_speak_windows, _speak_unix, _speak_pyttsx3):
        try:
            if backend(spoken, timeout):
                return True
        except Exception:
            logger.debug("tts backend %s failed", backend.__name__, exc_info=True)
    logger.info("no text-to-speech backend available; output stays text-only")
    return False


def _speak_windows(text: str, timeout: float) -> bool:
    if sys.platform != "win32":
        return False
    # The text is passed as base64 so no quote/newline in it can alter the
    # PowerShell command (the payload is only [A-Za-z0-9+/=]).
    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$t = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{payload}'));"
        "$s.Speak($t); $s.Dispose()"
    )
    result = subprocess.run(  # noqa: S603
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],  # noqa: S607
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return result.returncode == 0


def _speak_unix(text: str, timeout: float) -> bool:
    if sys.platform == "win32":
        return False
    command = "say" if sys.platform == "darwin" else "espeak"
    if shutil.which(command) is None:
        return False
    result = subprocess.run(  # noqa: S603
        [command, text], capture_output=True, timeout=timeout, check=False
    )
    return result.returncode == 0


def _speak_pyttsx3(text: str, _timeout: float) -> bool:
    import pyttsx3

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return True


def _first_message_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    content = getattr(getattr(choices[0], "message", None), "content", None)
    if isinstance(content, str):
        return content
    # Some providers return a list of content parts.
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""
