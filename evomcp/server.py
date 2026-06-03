"""evomcp MCP server.

Exposes GEPA + EvoX + Hybrid evolution as Claude MCP tools.

Run:
    evomcp serve           # stdio (default, works with Claude Code)
    evomcp serve --http    # HTTP/SSE on port 8765

Requires: pip install fastmcp
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run registry: in-process map from run_id → {status, config, pid, ...}
# Persisted to artifacts/runs/.registry.json for MCP-server restarts.
# ---------------------------------------------------------------------------

_REGISTRY_PATH = Path(os.environ.get("EVOMCP_ARTIFACTS", "artifacts")) / ".registry.json"


def _load_registry() -> dict[str, Any]:
    if _REGISTRY_PATH.exists():
        try:
            return json.loads(_REGISTRY_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_registry(reg: dict) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(reg, indent=2, default=str))


def _registry_add(run_id: str, entry: dict) -> None:
    reg = _load_registry()
    reg[run_id] = entry
    _save_registry(reg)


def _registry_get(run_id: str) -> dict | None:
    return _load_registry().get(run_id)


# ---------------------------------------------------------------------------
# Tool implementations (MCP-independent — also used by CLI)
# ---------------------------------------------------------------------------

def tool_evolve_run(
    mode: str,
    config_path: str,
    *,
    resume: bool = False,
    background: bool = True,
) -> dict[str, Any]:
    """Start an evolution run.

    Args:
        mode: "gepa", "evox", or "hybrid"
        config_path: path to a YAML config (relative to cwd or absolute)
        resume: continue from an existing checkpoint
        background: run in background (True) or block until complete (False)

    Returns:
        dict with run_id, mode, config_path, status ("started"|"complete"|"error")
    """
    if mode not in ("gepa", "evox", "hybrid"):
        return {"error": f"unknown mode {mode!r}; must be gepa, evox, or hybrid"}

    cfg = Path(config_path)
    if not cfg.exists():
        return {"error": f"config not found: {cfg}"}

    run_id = f"{mode}-{uuid.uuid4().hex[:8]}"
    runner_module = {
        "gepa":   "evomcp.optim.gepa_runner",
        "evox":   "evomcp.optim.evox_runner",
        "hybrid": "evomcp.optim.hybrid_runner",
    }[mode]

    cmd = [sys.executable, "-m", runner_module, str(cfg.resolve())]
    if resume:
        cmd.append("--resume")

    artifact_dir = Path(os.environ.get("EVOMCP_ARTIFACTS", "artifacts")) / "runs" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = artifact_dir / "run.log"

    entry: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "config_path": str(cfg.resolve()),
        "resume": resume,
        "artifact_dir": str(artifact_dir),
        "log": str(stdout_log),
        "status": "started",
        "pid": None,
    }

    if background:
        with open(stdout_log, "w") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        entry["pid"] = proc.pid
        entry["status"] = "running"
        _registry_add(run_id, entry)
        return {
            "run_id": run_id,
            "status": "running",
            "log": str(stdout_log),
            "hint": f"tail -f {stdout_log} to watch progress",
        }
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        entry["status"] = "complete" if result.returncode == 0 else "error"
        entry["returncode"] = result.returncode
        _registry_add(run_id, entry)
        return {
            "run_id": run_id,
            "status": entry["status"],
            "stdout_tail": result.stdout[-2000:],
            "stderr_tail": result.stderr[-1000:],
        }


def tool_evolve_status(run_id: str | None = None) -> dict[str, Any]:
    """Show status of active/recent runs.

    Args:
        run_id: specific run to query; None → all recent runs

    Returns:
        dict with runs list and best-so-far per run
    """
    registry = _load_registry()
    if run_id:
        entry = registry.get(run_id)
        if not entry:
            return {"error": f"run {run_id!r} not found"}
        return _enrich_entry(entry)

    runs = []
    for rid, entry in sorted(registry.items(), key=lambda x: x[0]):
        runs.append(_enrich_entry(entry))
    return {"runs": runs, "total": len(runs)}


def _enrich_entry(entry: dict) -> dict:
    """Add live best-so-far from pareto_archive.json / text_pareto_archive.json."""
    enriched = dict(entry)
    artifact_dir = Path(entry.get("artifact_dir", ""))

    # Check if process is still running
    pid = entry.get("pid")
    if pid and entry.get("status") == "running":
        try:
            os.kill(pid, 0)  # 0 = just check, don't kill
        except ProcessLookupError:
            enriched["status"] = "complete"

    for archive_name in ("pareto_archive.json", "text_pareto_archive.json"):
        ap = artifact_dir / archive_name
        if ap.exists():
            try:
                data = json.loads(ap.read_text())
                if data:
                    best = max(data, key=lambda x: x.get("result", {}).get("primary_score", float("-inf")))
                    enriched[f"best_{archive_name.replace('.json', '')}"] = {
                        "primary_score": best.get("result", {}).get("primary_score"),
                        "candidate_id": best.get("candidate", {}).get("candidate_id", "")[:12],
                    }
            except Exception:
                pass
    return enriched


def tool_evolve_export(run_id: str) -> dict[str, Any]:
    """Export the best candidate from a run as a replayable artifact.

    Writes artifacts/best/<run_id>/best_candidate.json and returns it.

    Args:
        run_id: ID of the run to export

    Returns:
        dict with best candidate + eval result
    """
    entry = _registry_get(run_id)
    if not entry:
        return {"error": f"run {run_id!r} not found"}

    artifact_dir = Path(entry["artifact_dir"])
    best_dir = Path(os.environ.get("EVOMCP_ARTIFACTS", "artifacts")) / "best" / run_id
    best_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"run_id": run_id, "exported_to": str(best_dir)}

    for archive_name, key in [
        ("pareto_archive.json", "prog"),
        ("text_pareto_archive.json", "text"),
    ]:
        ap = artifact_dir / archive_name
        if not ap.exists():
            continue
        data = json.loads(ap.read_text())
        if not data:
            continue
        best = max(data, key=lambda x: x.get("result", {}).get("primary_score", float("-inf")))
        out_path = best_dir / f"best_{key}_candidate.json"
        out_path.write_text(json.dumps(best, indent=2, default=str))
        result[f"best_{key}"] = {
            "primary_score": best.get("result", {}).get("primary_score"),
            "candidate_id": best.get("candidate", {}).get("candidate_id", "")[:12],
            "path": str(out_path),
        }

    return result


def tool_evolve_list_slots(project_root: str = ".") -> dict[str, Any]:
    """List registered text and prog slots from a project's search spaces.

    Args:
        project_root: directory containing optim/search_spaces/ to import

    Returns:
        dict with text_slots and prog_slots
    """
    project_root = str(Path(project_root).resolve())
    sys.path.insert(0, project_root)
    try:
        import importlib
        from evomcp.pipeline.registry import DEFAULT_REGISTRY

        # `optim.search_spaces` is project-local. The initial implementation
        # imported `evomcp.optim.search_spaces`, which makes `--project` a no-op.
        # Clear the registry and import the project's package fresh so repeated
        # CLI/MCP calls remain deterministic in one server process.
        DEFAULT_REGISTRY.text_slots.clear()
        DEFAULT_REGISTRY.prog_slots.clear()
        DEFAULT_REGISTRY._patch_env.clear()
        for mod in list(sys.modules):
            if mod == "optim" or mod.startswith("optim.search_spaces"):
                sys.modules.pop(mod, None)
        importlib.invalidate_caches()
        importlib.import_module("optim.search_spaces")

        return {
            "text_slots": {
                name: {
                    "role": s.role,
                    "description": s.description,
                    "mutatable": s.mutatable,
                    "max_chars": s.max_chars,
                }
                for name, s in DEFAULT_REGISTRY.text_slots.items()
            },
            "prog_slots": {
                name: {
                    "kind": s.kind.value,
                    "default": s.default,
                    "bounds": s.bounds,
                    "choices": list(s.choices) if s.choices else None,
                    "log_scale": s.log_scale,
                    "description": s.description,
                }
                for name, s in DEFAULT_REGISTRY.prog_slots.items()
            },
        }
    finally:
        if project_root in sys.path:
            sys.path.remove(project_root)


def tool_evolve_inspect(bundle_dir: str) -> dict[str, Any]:
    """Inspect a trace bundle for debugging / GEPA reflection.

    Args:
        bundle_dir: path to a trace bundle directory

    Returns:
        Compact feedback summary (success, score, errors, event count)
    """
    from evomcp.pipeline.tracing import read_bundle, trace_feedback_summary
    bd = Path(bundle_dir)
    if not bd.exists():
        return {"error": f"bundle dir not found: {bd}"}
    try:
        bundle = read_bundle(bd)
        summary = trace_feedback_summary(bd)
        return {
            "summary": summary,
            "candidate_id": bundle.get("candidate", {}).get("candidate_id", "")[:12],
            "stage": bundle.get("result", {}).get("stage"),
            "evaluator_version": bundle.get("result", {}).get("evaluator_version"),
            "git_sha": bundle.get("runtime_snapshot", {}).get("git_sha", "")[:8],
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

def build_server():
    """Build and return the FastMCP server instance."""
    try:
        from fastmcp import FastMCP
    except ImportError as e:
        raise ImportError(
            "fastmcp is required: pip install fastmcp>=2.0"
        ) from e

    mcp = FastMCP(
        "evomcp",
        instructions=(
            "GEPA + EvoX evolutionary optimizer. "
            "Use evolve_run to start an optimization, evolve_status to check progress, "
            "evolve_export to get the best candidate, evolve_inspect to read a trace bundle, "
            "evolve_list_slots to see what the optimizer can mutate."
        ),
    )

    @mcp.tool()
    def evolve_run(
        mode: str,
        config_path: str,
        resume: bool = False,
        background: bool = True,
    ) -> str:
        """Start a GEPA, EvoX, or hybrid evolution run.

        Args:
            mode: "gepa" (prompt evolution), "evox" (program evolution),
                  or "hybrid" (outer EvoX + inner GEPA)
            config_path: path to YAML config file
            resume: continue from existing checkpoint
            background: run in background (True) or block until complete

        Returns:
            JSON with run_id and status
        """
        return json.dumps(tool_evolve_run(mode, config_path, resume=resume, background=background), indent=2)

    @mcp.tool()
    def evolve_status(run_id: str = "") -> str:
        """Show status and best-so-far for active/recent runs.

        Args:
            run_id: specific run ID, or empty for all runs

        Returns:
            JSON with runs list or single run status
        """
        return json.dumps(tool_evolve_status(run_id or None), indent=2)

    @mcp.tool()
    def evolve_export(run_id: str) -> str:
        """Export the best candidate from a completed run.

        Args:
            run_id: ID of the run to export

        Returns:
            JSON with best candidate paths and scores
        """
        return json.dumps(tool_evolve_export(run_id), indent=2)

    @mcp.tool()
    def evolve_list_slots(project_root: str = ".") -> str:
        """List registered text and prog slots for a project.

        Args:
            project_root: directory containing optim/search_spaces/

        Returns:
            JSON with text_slots and prog_slots
        """
        return json.dumps(tool_evolve_list_slots(project_root), indent=2)

    @mcp.tool()
    def evolve_inspect(bundle_dir: str) -> str:
        """Inspect a trace bundle for debugging or GEPA reflection.

        Args:
            bundle_dir: path to a trace bundle directory

        Returns:
            JSON feedback summary
        """
        return json.dumps(tool_evolve_inspect(bundle_dir), indent=2)

    return mcp


def serve(http: bool = False, port: int = 8765) -> None:
    """Start the MCP server."""
    mcp = build_server()
    if http:
        mcp.run(transport="sse", host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
