"""Trace bundle writer + replay helpers.

parameter-golf showed that a single JSONL file is insufficient: GEPA needs
to hand the LM a richer trace with separate candidate, inputs, stdout, result
— and the evaluator needs a replay.json it can load cold.

This module implements the **bundle** model from parameter-golf:

  artifacts/runs/<run_id>/traces/<cid[:12]>-seed<s>/
    candidate.json          Full Candidate payload
    budget.json             Budget applied (including env_overrides)
    runtime_snapshot.json   Git SHA, Python, platform, pip hash, model versions
    inputs.json             Materialized text + prog genome fed to eval
    events.jsonl            Line-delimited events (command, tool_calls, ...)
    stdout.log              Evaluator subprocess stdout
    stderr.log              Evaluator subprocess stderr
    result.json             Full EvalResult (written at close)
    replay.json             Minimal replay manifest
    failure.json            Only if error_type is not None

TraceBundle is the write-side API.
read_bundle() is the read-side API (used by GEPA reflection).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evomcp.pipeline.candidate import Budget, Candidate, EvalResult


@dataclass
class RuntimeSnapshot:
    """Everything needed to replay or audit an evaluation."""

    git_sha: str = ""
    git_dirty: bool = False
    python_version: str = ""
    platform: str = ""
    dataset_version: str = ""
    evaluator_version: str = ""
    seed: int = 0
    env_hash: str = ""          # blake2b of `pip freeze` output
    model_versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        dataset_version: str,
        evaluator_version: str,
        seed: int,
        model_versions: dict[str, str] | None = None,
    ) -> "RuntimeSnapshot":
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            sha = "unknown"
        try:
            dirty_out = subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            )
            dirty = bool(dirty_out.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            dirty = False
        try:
            freeze = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
            )
            import hashlib as _h
            env_hash = _h.blake2b(freeze.encode(), digest_size=8).hexdigest()
        except Exception:
            env_hash = ""
        return cls(
            git_sha=sha,
            git_dirty=dirty,
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            dataset_version=dataset_version,
            evaluator_version=evaluator_version,
            seed=seed,
            env_hash=env_hash,
            model_versions=dict(model_versions or {}),
        )

    def to_dict(self) -> dict:
        return {
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "python_version": self.python_version,
            "platform": self.platform,
            "dataset_version": self.dataset_version,
            "evaluator_version": self.evaluator_version,
            "seed": self.seed,
            "env_hash": self.env_hash,
            "model_versions": self.model_versions,
        }


# Keep the old name as an alias so existing code doesn't break.
ReproContext = RuntimeSnapshot


class TraceBundle:
    """Write-side API for an evaluation trace bundle.

    Usage:
        bundle = TraceBundle.create(traces_dir, candidate, budget, snapshot)
        bundle.emit("command", {"cmd": "...", "cwd": "..."})
        # ... evaluation runs, writes stdout_path / stderr_path ...
        bundle.close(result, inputs={"text_genome": ..., "prog_env": ...})

    The bundle is not usable until close() is called (result.json missing).
    """

    def __init__(self, bundle_dir: Path):
        self.dir = bundle_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events_fp = open(self.dir / "events.jsonl", "w", encoding="utf-8")
        self._t0 = time.monotonic()
        # Pre-created paths callers may write to directly:
        self.stdout_path = self.dir / "stdout.log"
        self.stderr_path = self.dir / "stderr.log"

    @classmethod
    def create(
        cls,
        traces_dir: Path,
        candidate: Candidate,
        budget: Budget,
        snapshot: RuntimeSnapshot,
    ) -> "TraceBundle":
        """Create a new bundle directory and write static files immediately."""
        bundle_dir = (
            traces_dir
            / f"{candidate.candidate_id[:12]}-seed{snapshot.seed}"
        )
        bundle = cls(bundle_dir)
        _write_json(bundle_dir / "candidate.json", candidate.to_dict())
        _write_json(bundle_dir / "budget.json", budget.to_dict())
        _write_json(bundle_dir / "runtime_snapshot.json", snapshot.to_dict())
        return bundle

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        rec = {
            "t": round(time.monotonic() - self._t0, 4),
            "event": event,
            **payload,
        }
        self._events_fp.write(json.dumps(rec, default=str) + "\n")
        self._events_fp.flush()

    def close(
        self,
        result: EvalResult,
        *,
        inputs: dict[str, Any] | None = None,
    ) -> Path:
        """Finalise the bundle. Returns bundle_dir path.

        Writes:
          inputs.json     — materialized genomes + env (if provided)
          result.json     — full EvalResult
          replay.json     — minimal key-only manifest
          failure.json    — if result.error_type is not None
        """
        self._events_fp.close()
        if inputs is not None:
            _write_json(self.dir / "inputs.json", inputs)
        _write_json(self.dir / "result.json", result.to_dict())
        _write_json(self.dir / "replay.json", {
            "candidate_id": result.candidate_id,
            "stage": result.stage,
            "seed": result.seed,
            "dataset_version": result.dataset_version,
            "evaluator_version": result.evaluator_version,
            "success": result.success,
            "primary_score": result.primary_score,
        })
        if result.error_type is not None:
            _write_json(self.dir / "failure.json", {
                "error_type": result.error_type.value,
                "error_message": result.error_message,
            })
        return self.dir

    def __enter__(self) -> "TraceBundle":
        return self

    def __exit__(self, *exc) -> None:
        if not self._events_fp.closed:
            self._events_fp.close()


# ---------------------------------------------------------------------------
# Backward-compat single-file writer (kept for small scripts)
# ---------------------------------------------------------------------------

class TraceWriter:
    """Legacy JSONL writer — use TraceBundle for full evaluations."""

    def __init__(self, trace_path: Path, repro: RuntimeSnapshot | None = None):
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = trace_path
        self._fp = open(trace_path, "w", encoding="utf-8")
        self._t0 = time.monotonic()
        if repro is not None:
            self.emit("repro", repro.to_dict())

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        rec = {"t": round(time.monotonic() - self._t0, 4), "event": event, **payload}
        self._fp.write(json.dumps(rec, default=str) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Read-side helpers
# ---------------------------------------------------------------------------

def read_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    """Load a trace bundle back into memory.

    Returns a dict with keys: candidate, budget, snapshot, inputs, events,
    result, replay, failure (if present).
    """
    bd = Path(bundle_dir)
    out: dict[str, Any] = {}
    for name in ("candidate", "budget", "runtime_snapshot", "inputs",
                 "result", "replay", "failure"):
        p = bd / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    events_p = bd / "events.jsonl"
    if events_p.exists():
        out["events"] = [
            json.loads(line)
            for line in events_p.read_text().splitlines()
            if line.strip()
        ]
    return out


def read_trace(path: Path | str) -> list[dict]:
    """Load a legacy single-file trace."""
    out = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def trace_feedback_summary(bundle_dir: Path | str) -> dict[str, Any]:
    """Compact summary of an evaluation bundle, suitable for GEPA reflection.

    Returns a dict with: success, primary_score, secondary_scores,
    error_type, tool_call_count, event_count.
    """
    bd = read_bundle(bundle_dir)
    result = bd.get("result", {})
    events = bd.get("events", [])
    return {
        "success": result.get("success", False),
        "primary_score": result.get("primary_score", None),
        "secondary_scores": result.get("secondary_scores", {}),
        "error_type": result.get("error_type"),
        "error_message": result.get("error_message"),
        "tool_call_count": sum(1 for e in events if e.get("event") == "tool_call"),
        "event_count": len(events),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
