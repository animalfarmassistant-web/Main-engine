# Vendored: Graphify

This directory is a **verbatim copy** of the `graphify` package, vendored into
VibeAssistant so the `strix vibe` prompt-orchestrator is fully self-contained
(no external `graphify` CLI or PyPI install required at runtime).

- **Upstream:** https://github.com/Graphify-Labs/graphify
- **Version:** 0.9.35
- **Commit:** 9f25a3aaa1050913c2d8a1b9f0b0f0ed18296abd
- **Vendored on:** 2026-08-07
- **License:** Apache-2.0 with MIT-licensed portions — see `LICENSE`, `LICENSE-MIT`,
  and `NOTICE` in this directory. Copyright 2026 Safi Shamsi and the Graphify
  contributors. Attribution is preserved.

## How it is loaded

The package keeps its original top-level import name `graphify` (329 internal
`import graphify.*` statements are left untouched). `strix/vibe/graph.py`
prepends this `_vendor/` directory to `sys.path` before importing, so
`import graphify` resolves here. Do **not** rename the inner `graphify/`
directory or rewrite its internal imports.

## What VibeAssistant uses

Only the importable indexing pipeline is called:
`graphify.collect_files`, `graphify.extract`, `graphify.build_from_json`,
`graphify.to_json` (and optionally `graphify.cluster`). Retrieval
(query/path/explain) is implemented in `strix/vibe/graph.py` over the emitted
`graph.json` — Graphify's own query lives in its CLI layer and is not imported.

## Updating

Re-clone upstream at the desired commit, replace the inner `graphify/`
directory wholesale, keep the license files + this VENDOR.md, and bump the
commit/version above. Re-run `tests/vibe/` afterwards.
