"""Shared LLM plumbing for the vibe orchestrator.

A single ``run_completion`` used by both the prompt *polisher* (tightens an
assembled brief) and the prompt *synthesizer* (authors a detailed engineering
prompt). It reuses Strix's model stack — ``StrixProvider`` + LiteLLM routing —
so the same ``STRIX_LLM`` / ``OLLAMA_API_BASE`` config drives scans and vibe.
No sandbox, no tools: one system + user turn, text back.
"""

from __future__ import annotations

from typing import Any


class NoModelConfiguredError(RuntimeError):
    """Raised when a vibe LLM feature runs without ``STRIX_LLM`` set."""

    def __init__(self) -> None:
        super().__init__(
            "No LLM configured. Set STRIX_LLM (e.g. ollama/llama3.1) and, for a "
            "local runner, OLLAMA_API_BASE / LLM_API_BASE (or LLM_API_KEY for a "
            "hosted model)."
        )


async def run_completion(
    system_instructions: str,
    user_input: str,
    *,
    reasoning_effort: str | None = None,
) -> str:
    """Run one non-streaming system+user completion and return the text.

    ``reasoning_effort`` overrides the configured effort for this call — pass
    ``"none"`` for latency-sensitive, short-answer work (e.g. voice chat), where
    a reasoning model's long deliberation is pure cost.

    Raises :class:`NoModelConfiguredError` if no model is configured, and
    propagates provider errors to the caller (callers decide whether to treat
    the feature as best-effort or hard-fail).
    """
    from strix.config import load_settings

    settings = load_settings()
    model_name = (settings.llm.model or "").strip()
    if not model_name:
        # Fail before importing the heavy agents/litellm stack just to error out.
        raise NoModelConfiguredError

    from agents.models.interface import ModelTracing

    from strix.config.models import StrixProvider, configure_sdk_model_defaults
    from strix.core.inputs import make_model_settings

    configure_sdk_model_defaults(settings)
    model = StrixProvider().get_model(model_name)
    model_settings = make_model_settings(
        reasoning_effort or settings.llm.reasoning_effort,  # type: ignore[arg-type]
        model_name=model_name,
        request_timeout=settings.llm.timeout,
        prompt_cache=False,
    )
    response = await model.get_response(
        system_instructions=system_instructions,
        input=user_input,
        model_settings=model_settings,
        tools=[],
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )
    return extract_text(response).strip()


def extract_text(response: Any) -> str:
    """Pull assistant text out of a ModelResponse, tolerating shape drift."""
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        content = getattr(item, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for chunk in content:
                text = getattr(chunk, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)
