"""Write guardrail for the read-only ``vibe`` prompt orchestrator.

The vibe mode analyses the user's project but must NEVER write inside it. Its
only legitimate write targets are the vibe state dir (``.vibe/``) and any
``graphify-out`` directory (the vendored Graphify engine's own artifact dir).
Every file the vibe path creates routes through :func:`assert_write_allowed`,
so a stray write into the source tree fails loudly instead of silently
mutating the user's code.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


class VibeWriteError(RuntimeError):
    """Raised when the vibe path attempts a write outside its allowed roots."""


def _resolve(path: str | Path) -> Path:
    # ``strict=False`` so we can validate a target that does not exist yet
    # (the common case: we are about to create it).
    return Path(path).expanduser().resolve(strict=False)


def is_within(path: str | Path, root: str | Path) -> bool:
    """Return whether *path* is *root* itself or nested under it."""
    resolved = _resolve(path)
    resolved_root = _resolve(root)
    return resolved == resolved_root or resolved_root in resolved.parents


def assert_write_allowed(target: str | Path, allowed_roots: Iterable[str | Path]) -> Path:
    """Return the resolved *target* if it is inside one of *allowed_roots*, else raise.

    Args:
        target: The file or directory the caller intends to write.
        allowed_roots: Directories writes may land in (e.g. the ``.vibe`` state
            dir). A target equal to or nested under any of these passes.

    Raises:
        VibeWriteError: If *target* is not within any allowed root — meaning the
            caller was about to write into the user's project or elsewhere.
    """
    resolved = _resolve(target)
    roots = [_resolve(root) for root in allowed_roots]
    if not roots:
        raise VibeWriteError(
            f"Refusing to write {resolved}: no allowed write roots were configured."
        )
    if any(resolved == root or root in resolved.parents for root in roots):
        return resolved
    allowed = ", ".join(str(root) for root in roots)
    raise VibeWriteError(
        f"Refusing to write outside the vibe workspace.\n"
        f"  target : {resolved}\n"
        f"  allowed: {allowed}\n"
        "The vibe prompt orchestrator is read-only w.r.t. your project; it may "
        "only write under .vibe/ (or a graphify-out/ dir)."
    )


def safe_mkdir(target: str | Path, allowed_roots: Iterable[str | Path]) -> Path:
    """Create *target* (and parents) after a write-guard check; return it resolved."""
    resolved = assert_write_allowed(target, allowed_roots)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_write_text(
    target: str | Path,
    text: str,
    allowed_roots: Iterable[str | Path],
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write *text* to *target* after a write-guard check.

    Uses a same-directory temp file + :func:`os.replace` so a reader never sees a
    half-written file (the pattern Strix's notes store uses in ``_persist``).
    """
    resolved = assert_write_allowed(target, allowed_roots)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        dir=str(resolved.parent),
        prefix=f".{resolved.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(resolved)
    return resolved
