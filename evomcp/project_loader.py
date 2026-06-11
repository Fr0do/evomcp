"""Helpers for loading project-specific search spaces into the shared registry."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from evomcp.pipeline.registry import DEFAULT_REGISTRY


def infer_project_root(path_hint: str | Path) -> Path:
    """Infer the project root that owns `optim/search_spaces`.

    `path_hint` may be a config file path or an explicit project directory.
    """
    candidate = Path(path_hint).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent

    search_roots = [candidate, *candidate.parents]
    for root in search_roots:
        if (root / "optim" / "search_spaces").is_dir():
            return root
    return candidate


def load_project_slots(path_hint: str | Path) -> Path:
    """Load `optim.search_spaces` from a project root into DEFAULT_REGISTRY."""
    project_root = infer_project_root(path_hint)

    DEFAULT_REGISTRY.reset()

    for name in list(sys.modules):
        if name == "optim" or name.startswith("optim."):
            sys.modules.pop(name, None)

    sys.path.insert(0, str(project_root))
    try:
        importlib.invalidate_caches()
        importlib.import_module("optim.search_spaces")
    finally:
        try:
            sys.path.remove(str(project_root))
        except ValueError:
            pass

    return project_root
