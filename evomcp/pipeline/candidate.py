"""Canonical Candidate and EvalResult dataclasses.

These are the single source of truth across GEPA, EvoX, and hybrid modes.
Any optimizer that doesn't speak this protocol cannot be integrated.

Insights merged from parameter-golf:
- Budget carries env_overrides (arbitrary evaluator knobs per stage) and
  a stable_hash() used in cache-key derivation.
- Candidate exposes stable_hash() (all fields, not just genomes) for easy
  dedup across runs.
- EvalResult carries a trace_bundle_dir (directory) rather than separate
  path fields; individual artefacts live inside the bundle.
- failure_penalty is now a top-level constant so scorers stay consistent.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

# Penalized score for invalid / failed candidates.
# Must be lower than any valid score so selection is straightforward.
FAILURE_PENALTY: float = -1_000_000.0


class FailureClass(str, Enum):
    """Failure taxonomy. GEPA reflects on BAD_FORMAT/LOW_SCORE; EvoX uses all."""

    SYNTAX_BUILD = "syntax_build"
    RUNTIME = "runtime"
    TIMEOUT = "timeout"
    OOM = "oom"
    BAD_FORMAT = "bad_format"
    LOW_SCORE = "low_score"
    JUDGE_FAIL = "judge_fail"
    TOOL_FAIL = "tool_fail"


@dataclass(frozen=True)
class CostMetrics:
    usd: float = 0.0
    wall_s: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def _stable_hash(obj: Any, digest_size: int = 8) -> str:
    """Deterministic hash of any JSON-serializable object."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.blake2b(blob, digest_size=digest_size).hexdigest()


@dataclass(frozen=True)
class Budget:
    """Per-evaluation budget.

    env_overrides lets each stage inject arbitrary key-value pairs into the
    evaluation subprocess (e.g. ITERATIONS=50, MAX_WALLCLOCK=120 for smoke).
    This mirrors parameter-golf's BudgetSpec.env_overrides so the evaluator
    can cap expensive runs without hard-coding stage logic.

    stable_hash() is used as part of the cache key so a budget change
    (e.g. increasing max_judges from 1 to 3) automatically invalidates the
    cache for that stage.
    """

    stage: int                              # 0=smoke, 1=proxy, 2=subset, 3=full
    max_plans: int
    max_judges: int
    timeout_s: int
    max_usd: float | None = None
    env_overrides: dict[str, Any] = field(default_factory=dict)

    def stable_hash(self) -> str:
        return _stable_hash({
            "stage": self.stage,
            "max_plans": self.max_plans,
            "max_judges": self.max_judges,
            "timeout_s": self.timeout_s,
            "max_usd": self.max_usd,
            "env_overrides": dict(self.env_overrides),
        })

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "max_plans": self.max_plans,
            "max_judges": self.max_judges,
            "timeout_s": self.timeout_s,
            "max_usd": self.max_usd,
            "env_overrides": dict(self.env_overrides),
        }


@dataclass(frozen=True)
class Candidate:
    """A point in the joint (text_genome × prog_genome) search space.

    candidate_id is derived from both genomes so equality is content-based.
    parent_ids lets us reconstruct the evolutionary lineage for archive analysis.
    stable_hash() covers the full object including metadata (used for dedup).
    """

    text_genome: Mapping[str, str] = field(default_factory=dict)
    prog_genome: Mapping[str, Any] = field(default_factory=dict)
    parent_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            cid = _stable_hash(
                {"text": dict(self.text_genome), "prog": dict(self.prog_genome)}
            )
            object.__setattr__(self, "candidate_id", cid)

    def text_hash(self) -> str:
        return _stable_hash(dict(self.text_genome))

    def prog_hash(self) -> str:
        return _stable_hash(dict(self.prog_genome))

    def stable_hash(self) -> str:
        """Full hash including parent_ids + metadata. Use for run-level dedup."""
        return _stable_hash(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "text_genome": dict(self.text_genome),
            "prog_genome": dict(self.prog_genome),
            "parent_ids": list(self.parent_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Candidate":
        return cls(
            text_genome=dict(d.get("text_genome", {})),
            prog_genome=dict(d.get("prog_genome", {})),
            parent_ids=tuple(d.get("parent_ids", [])),
            metadata=dict(d.get("metadata", {})),
            candidate_id=str(d.get("candidate_id", "")),
        )


@dataclass
class EvalResult:
    """Canonical evaluator output.

    trace_bundle_dir points to the evaluation's artefact directory
    (written by pipeline.tracing.TraceBundle). Individual files inside:
      candidate.json, budget.json, runtime_snapshot.json, inputs.json,
      events.jsonl, stdout.log, stderr.log, result.json, replay.json,
      failure.json (if !success).

    A candidate is not valid for export unless it has a complete EvalResult
    with a populated trace_bundle_dir that survives disk round-trips.
    """

    candidate_id: str
    success: bool
    primary_score: float
    secondary_scores: dict[str, float] = field(default_factory=dict)
    cost: CostMetrics = field(default_factory=CostMetrics)
    trace_bundle_dir: Path | None = None    # replaces three separate path fields
    error_type: FailureClass | None = None
    error_message: str | None = None
    evaluator_version: str = ""
    seed: int = 0
    dataset_version: str = ""
    stage: int = 0
    cache_hit: bool = False                 # True when this result came from cache

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cost"] = asdict(self.cost)
        d["error_type"] = self.error_type.value if self.error_type else None
        d["trace_bundle_dir"] = str(self.trace_bundle_dir) if self.trace_bundle_dir else None
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EvalResult":
        cost_d = d.get("cost", {})
        return cls(
            candidate_id=str(d["candidate_id"]),
            success=bool(d["success"]),
            primary_score=float(d["primary_score"]),
            secondary_scores=dict(d.get("secondary_scores", {})),
            cost=CostMetrics(**{k: cost_d[k] for k in CostMetrics.__dataclass_fields__ if k in cost_d}),
            trace_bundle_dir=Path(d["trace_bundle_dir"]) if d.get("trace_bundle_dir") else None,
            error_type=FailureClass(d["error_type"]) if d.get("error_type") else None,
            error_message=d.get("error_message"),
            evaluator_version=d.get("evaluator_version", ""),
            seed=int(d.get("seed", 0)),
            dataset_version=d.get("dataset_version", ""),
            stage=int(d.get("stage", 0)),
            cache_hit=bool(d.get("cache_hit", False)),
        )

    @classmethod
    def penalized(
        cls,
        candidate_id: str,
        error_type: FailureClass,
        message: str,
        *,
        stage: int = 0,
        evaluator_version: str = "",
        seed: int = 0,
        dataset_version: str = "",
    ) -> "EvalResult":
        """Build a penalized result for invalid / failed candidates.

        Invalid candidates must not crash the run; we return a FAILURE_PENALTY
        score with a populated error_type so EvoX can select against them and
        GEPA can decide whether to reflect (it should only reflect on
        BAD_FORMAT and LOW_SCORE — semantically informative failures).
        """
        return cls(
            candidate_id=candidate_id,
            success=False,
            primary_score=FAILURE_PENALTY,
            error_type=error_type,
            error_message=message,
            stage=stage,
            evaluator_version=evaluator_version,
            seed=seed,
            dataset_version=dataset_version,
        )
