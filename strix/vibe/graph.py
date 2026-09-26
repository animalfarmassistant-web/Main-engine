"""Read-only project indexing + retrieval for the vibe orchestrator.

Indexing uses the **vendored** Graphify engine (``strix/vibe/_vendor/graphify``)
via direct imports: ``collect_files`` -> ``extract`` -> ``build_from_json`` ->
``to_json``. All output is routed under ``.vibe/`` so the user's source tree is
never modified. If Graphify's native parser stack (tree-sitter) is unavailable,
a pure-Python :mod:`ast` walker produces a compatible minimal graph so the tool
still works offline.

Retrieval (``query`` / ``explain``) is implemented here over the emitted
``graph.json`` — Graphify's own query lives in its CLI layer and is not imported.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from strix.vibe.paths import assert_write_allowed, safe_mkdir, safe_write_text


if TYPE_CHECKING:
    from strix.vibe.config import VibeSettings


logger = logging.getLogger(__name__)

_VENDOR_DIR = Path(__file__).parent / "_vendor"

# Directories we never descend into during the ast fallback walk.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".vibe",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".tox",
        "graphify-out",
    }
)


class GraphifyManager:
    """Builds and queries a read-only knowledge graph of the project."""

    def __init__(self, settings: VibeSettings) -> None:
        self._settings = settings
        self._graph_path = settings.graph_json_path

    # -- vendored engine loading ------------------------------------------
    def _load_graphify(self) -> SimpleNamespace | None:
        """Import the vendored graphify engine functions, or None if unusable.

        Functions are imported straight from their submodules rather than as
        package attributes: ``graphify.extract`` is both a lazy function export
        *and* a submodule, and once any function from that module loads, the
        submodule shadows the function on the package. Direct submodule imports
        sidestep that collision.

        ``GRAPHIFY_OUT`` is set before import because graphify reads it into a
        module-level constant; pointing it under ``.vibe`` keeps every default
        artifact path inside the vibe workspace.
        """
        os.environ.setdefault("GRAPHIFY_OUT", str(self._settings.graph_out_dir))
        vendor = str(_VENDOR_DIR)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        try:
            from graphify.build import build_from_json  # noqa: PLC0415
            from graphify.export import to_json  # noqa: PLC0415
            from graphify.extract import collect_files, extract  # noqa: PLC0415
        except Exception:
            logger.warning("vendored graphify unavailable; using ast fallback", exc_info=True)
            return None
        try:
            from graphify.cluster import cluster  # noqa: PLC0415
        except Exception:  # noqa: BLE001 - clustering is optional enrichment.
            cluster = None  # type: ignore[assignment]
        return SimpleNamespace(
            collect_files=collect_files,
            extract=extract,
            build_from_json=build_from_json,
            to_json=to_json,
            cluster=cluster,
        )

    # -- existence check -----------------------------------------------------
    def has_source_files(self) -> bool:
        """Whether ``project_root`` contains anything worth indexing.

        Cheap and language-agnostic: no graphify/ast run, just a short-circuit
        walk for the first non-noise file. Callers use this to decide between
        "discuss the existing project" and "no project yet -> planning mode"
        *before* spending time on a full index.
        """
        root = self._settings.project_root
        if not root.is_dir():
            return False
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            current = Path(dirpath)
            for name in filenames:
                rel = (current / name).relative_to(root).as_posix()
                if not _is_noise_path(rel):
                    return True
        return False

    # -- indexing ----------------------------------------------------------
    def ensure_index(self, *, force: bool = False) -> Path:
        """Build ``graph.json`` if missing (or *force*); return its path."""
        if self._graph_path.exists() and not force:
            logger.info("reusing existing graph at %s", self._graph_path)
            return self._graph_path

        allowed = self._settings.allowed_write_roots
        safe_mkdir(self._settings.graph_out_dir, allowed)

        graphify = self._load_graphify()
        if graphify is not None:
            try:
                return self._build_with_graphify(graphify)
            except Exception:
                logger.warning(
                    "graphify indexing failed; falling back to ast walker", exc_info=True
                )
        return self._build_with_ast()

    def _build_with_graphify(self, graphify: Any) -> Path:
        root = self._settings.project_root
        all_files = list(graphify.collect_files(root, root=root))
        files = [f for f in all_files if not self._is_noise_file(f, root)]
        logger.info(
            "graphify: extracting %d of %d file(s) under %s (%d noise skipped)",
            len(files),
            len(all_files),
            root,
            len(all_files) - len(files),
        )
        extraction = graphify.extract(files, cache_root=self._settings.graph_out_dir, root=root)
        graph = graphify.build_from_json(extraction, root=root)
        try:
            communities = graphify.cluster(graph)
        except Exception:  # noqa: BLE001 - clustering is optional enrichment.
            logger.info("graphify clustering unavailable; writing graph without communities")
            communities = {}
        # Guard the write target before handing the path to graphify.
        assert_write_allowed(self._graph_path, self._settings.allowed_write_roots)
        graphify.to_json(graph, communities, str(self._graph_path), force=True)
        logger.info("graphify: wrote graph to %s", self._graph_path)
        return self._graph_path

    # -- ast fallback ------------------------------------------------------
    def _build_with_ast(self) -> Path:
        root = self._settings.project_root
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def add_node(node_id: str, **attrs: Any) -> None:
            if node_id in seen_ids:
                return
            seen_ids.add(node_id)
            nodes.append({"id": node_id, **attrs})

        for py_file in self._iter_python_files(root):
            rel = py_file.relative_to(root).as_posix()
            add_node(rel, label=rel, kind="file", source_file=rel)
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except (OSError, SyntaxError, ValueError):
                continue
            for child in ast.iter_child_nodes(tree):
                if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                    kind = "class" if isinstance(child, ast.ClassDef) else "function"
                    sym_id = f"{rel}::{child.name}"
                    add_node(
                        sym_id,
                        label=child.name,
                        kind=kind,
                        source_file=rel,
                        line=child.lineno,
                    )
                    edges.append({"source": rel, "target": sym_id, "type": "contains"})
                elif isinstance(child, ast.Import | ast.ImportFrom):
                    for module in self._imported_modules(child):
                        add_node(module, label=module, kind="module", source_file=None)
                        edges.append({"source": rel, "target": module, "type": "imports"})

        graph = {
            "directed": False,
            "multigraph": False,
            "graph": {"builder": "vibe-ast-fallback"},
            "nodes": nodes,
            "edges": edges,
        }
        text = json.dumps(graph, ensure_ascii=False, indent=2)
        safe_write_text(self._graph_path, text, self._settings.allowed_write_roots)
        logger.info(
            "ast fallback: wrote graph with %d node(s), %d edge(s) to %s",
            len(nodes),
            len(edges),
            self._graph_path,
        )
        return self._graph_path

    def _iter_python_files(self, root: Path) -> list[Path]:
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            found.extend(Path(dirpath) / name for name in filenames if name.endswith(".py"))
        return found

    @staticmethod
    def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
        if isinstance(node, ast.Import):
            return [alias.name.split(".")[0] for alias in node.names]
        if node.module:
            return [node.module.split(".")[0]]
        return []

    # -- retrieval ---------------------------------------------------------
    def load_graph(self) -> dict[str, Any]:
        """Load the emitted graph.json (node_link shape); ``{}`` if absent/bad."""
        if not self._graph_path.exists():
            return {}
        try:
            data = json.loads(self._graph_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("graph.json at %s is unreadable", self._graph_path)
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _edges(graph: dict[str, Any]) -> list[dict[str, Any]]:
        # networkx <=3.1 serialises as "links"; newer as "edges".
        raw = graph.get("edges")
        if raw is None:
            raw = graph.get("links", [])
        return [e for e in raw if isinstance(e, dict)]

    def query(self, intent: str, *, limit: int = 12) -> list[dict[str, Any]]:
        """Return graph nodes most relevant to *intent*, with 1-hop neighbours.

        Keyword overlap against each node's (cleaned) label / id / source_file,
        boosted by graph centrality and a preference for code over docs. Noise
        (vendored code, build assets, Graphify's own ``rationale``/``concept``
        nodes) is dropped. For broad intents with no keyword hits, falls back to
        the most central code nodes so the real architecture surfaces.
        """
        graph = self.load_graph()
        all_nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
        if not all_nodes:
            return []

        node_by_id = {str(n.get("id")): n for n in all_nodes}
        degree = self._degree(graph)
        max_degree = max(degree.values(), default=1) or 1
        adjacency = self._build_adjacency(graph)

        kept = [n for n in all_nodes if self._is_useful_node(n)]
        terms = [t for t in _tokenize(intent) if len(t) > 2 and t not in _STOPWORDS]

        scored: list[tuple[float, dict[str, Any]]] = []
        for node in kept:
            nid = str(node.get("id"))
            haystack = f"{_clean_label(node)} {nid} {node.get('source_file') or ''}".lower()
            keywords = sum(haystack.count(term) for term in terms)
            if keywords == 0:
                continue
            centrality = degree.get(nid, 0) / max_degree
            code_boost = 1.5 if node.get("file_type") == "code" else 0.0
            scored.append((keywords * 10 + code_boost + centrality * 4, node))

        if scored:
            scored.sort(key=lambda pair: pair[0], reverse=True)
            chosen = scored[:limit]
        else:
            # Broad intent (e.g. "explain the architecture") — surface the most
            # central code nodes instead of returning nothing.
            code_nodes = [n for n in kept if n.get("file_type") == "code"]
            code_nodes.sort(key=lambda n: degree.get(str(n.get("id")), 0), reverse=True)
            chosen = [(float(degree.get(str(n.get("id")), 0)), n) for n in code_nodes[:limit]]

        results: list[dict[str, Any]] = []
        for score, node in chosen:
            node_id = str(node.get("id"))
            neighbours = [
                _clean_label(node_by_id[nb])
                for nb in sorted(adjacency.get(node_id, set()))
                if nb in node_by_id
            ][:6]
            results.append(
                {
                    "id": node_id,
                    "label": _clean_label(node),
                    "kind": _node_kind(node),
                    "language": _node_language(node),
                    "source_file": node.get("source_file"),
                    "line": node.get("source_location"),
                    "score": round(score, 2),
                    "neighbours": neighbours,
                }
            )
        return results

    @staticmethod
    def _is_noise_file(path: Path, root: Path) -> bool:
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.as_posix()
        return _is_noise_path(rel)

    @staticmethod
    def _is_useful_node(node: dict[str, Any]) -> bool:
        # Graphify emits LLM-written 'rationale'/'concept' prose nodes — noise for
        # prompts. And skip vendored/build-artifact paths.
        if node.get("file_type") in {"rationale", "concept"}:
            return False
        return not _is_noise_path(str(node.get("source_file") or ""))

    def _degree(self, graph: dict[str, Any]) -> dict[str, int]:
        degree: dict[str, int] = {}
        for edge in self._edges(graph):
            for endpoint in (edge.get("source"), edge.get("target")):
                key = str(endpoint)
                if key != "None":
                    degree[key] = degree.get(key, 0) + 1
        return degree

    def explain(self, node_id: str) -> dict[str, Any] | None:
        """Return a node plus its direct neighbours, or None if not present."""
        graph = self.load_graph()
        node_by_id = {
            str(n.get("id")): n for n in graph.get("nodes", []) if isinstance(n, dict)
        }
        node = node_by_id.get(node_id)
        if node is None:
            return None
        adjacency = self._build_adjacency(graph)
        neighbours = [
            {"id": nb, "label": node_by_id[nb].get("label", nb), "kind": node_by_id[nb].get("kind")}
            for nb in sorted(adjacency.get(node_id, set()))
            if nb in node_by_id
        ]
        return {"node": node, "neighbours": neighbours}

    def _build_adjacency(self, graph: dict[str, Any]) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = {}
        for edge in self._edges(graph):
            src = str(edge.get("source"))
            dst = str(edge.get("target"))
            if src == "None" or dst == "None":
                continue
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)
        return adjacency


def _tokenize(text: str) -> list[str]:
    token: list[str] = []
    out: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            token.append(ch)
        elif token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out


# Generic verbs/fillers that would otherwise match doc prose everywhere.
_STOPWORDS = frozenset(
    {
        "add", "the", "and", "for", "with", "want", "need", "make", "create",
        "implement", "please", "explain", "fix", "update", "change", "into",
        "from", "that", "this", "use", "using", "code", "feature", "logic",
        "support", "new", "get", "set", "run", "how", "why", "what",
    }
)

_GRAPHIFY_REF_PREFIX = "graphify reference: "
_LABEL_MAX = 60

# Vendored code, build artifacts, minified/hashed bundles — not the user's source.
_NOISE_DIR_PARTS = frozenset(
    {"_vendor", ".vibe", "node_modules", "dist", "build", "__pycache__", ".git"}
)
_NOISE_PATH_RE = re.compile(
    r"(^|/)static/assets/|\.min\.js$|(^|/)[^/]*-[0-9a-f]{8}\.[a-z]+$",
    re.IGNORECASE,
)


def _is_noise_path(source_file: str) -> bool:
    if not source_file:
        return False
    posix = source_file.replace("\\", "/")
    if _NOISE_DIR_PARTS & set(posix.split("/")):
        return True
    return _NOISE_PATH_RE.search(posix) is not None


def _node_metadata(node: dict[str, Any]) -> dict[str, Any]:
    """Graphify stores metadata as a stringified dict; parse it defensively."""
    raw = node.get("metadata")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _node_kind(node: dict[str, Any]) -> str:
    kind = _node_metadata(node).get("kind")
    if isinstance(kind, str) and kind:
        return kind
    file_type = node.get("file_type")
    if isinstance(file_type, str) and file_type:
        return file_type
    return str(node.get("kind") or "unknown")


def _node_language(node: dict[str, Any]) -> str | None:
    language = _node_metadata(node).get("language")
    return language if isinstance(language, str) and language else None


def _clean_label(node: dict[str, Any]) -> str:
    label = str(node.get("label") or node.get("norm_label") or node.get("id") or "?")
    if label.lower().startswith(_GRAPHIFY_REF_PREFIX):
        label = label[len(_GRAPHIFY_REF_PREFIX) :]
    label = " ".join(label.split())
    return label if len(label) <= _LABEL_MAX else f"{label[:_LABEL_MAX].rstrip()}…"
