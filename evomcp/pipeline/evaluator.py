"""Canonical Evaluator protocol + utilities.

One evaluator scores any Candidate (text-only, prog-only, or hybrid) so
GEPA and EvoX results are directly comparable.

Concrete evaluators live in `pipeline/evaluators/` (added in Step 4+).
The protocol here enforces the contract: deterministic under fixed seed,
writes a replayable trace bundle, returns a typed EvalResult.

Key patterns merged from parameter-golf:
- cache_key() uses Budget.stable_hash() so any budget change auto-invalidates.
- materialize_prog_genome() merges base + candidate + patch env_overrides +
  budget env_overrides into a flat dict the subprocess can consume.
- ScoringConfig captures weights and constraint penalties so all evaluators
  stay consistent (no magic numbers scattered in runner code).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from evomcp.pipeline.candidate import Budget, Candidate, EvalResult, FAILURE_PENALTY


@dataclass(frozen=True)
class ScoringConfig:
    """Weights and penalty scalars for primary-score computation.

    primary_score = Σ weight[k] * metric[k]
                  + constraint_penalty * Σ max(0, metric[k] - constraint[k])
                  (or FAILURE_PENALTY when success=False)

    Units matter: quality is a signed metric (higher is better),
    so use negative weights for cost metrics (latency, tokens, memory).
    """

    failure_penalty: float = FAILURE_PENALTY
    constraint_penalty: float = -0.001     # multiplied by the constraint violation amount
    weights: dict[str, float] = field(default_factory=lambda: {
        "primary_score": 1.0,
        "latency_s":     -0.01,
        "tokens":        -0.0001,
    })
    # Hard upper bounds per secondary metric. Violations add constraint_penalty
    # * excess.  Use for absolute limits (e.g. latency must stay < 30s).
    constraints: dict[str, float] = field(default_factory=dict)


def compute_primary_score(
    metrics: dict[str, float],
    cfg: ScoringConfig,
    *,
    success: bool,
) -> float:
    """Weighted-sum scoring from parameter-golf, adapted for HANFU metrics."""
    if not success:
        return cfg.failure_penalty
    score = 0.0
    for key, w in cfg.weights.items():
        score += w * metrics.get(key, 0.0)
    for key, bound in cfg.constraints.items():
        excess = metrics.get(key, 0.0) - bound
        if excess > 0:
            score += cfg.constraint_penalty * excess
    return score


@runtime_checkable
class Evaluator(Protocol):
    """All HANFU evaluators must satisfy this protocol."""

    version: str  # bumped whenever scoring semantics change; part of cache key

    def evaluate(
        self,
        candidate: Candidate,
        budget: Budget,
        seed: int,
        *,
        run_dir: Path,
    ) -> EvalResult:
        """Score a candidate.

        Args:
            candidate: text + program genome under evaluation.
            budget: stage, limits, timeout, env_overrides.
            seed: deterministic seed; same (candidate, budget, seed)
                must yield the same primary_score (± external-API noise).
            run_dir: parent directory; evaluator writes a trace bundle
                under run_dir/traces/{candidate_id[:12]}-seed{seed}/.

        Returns:
            EvalResult with success flag, primary + secondary scores, cost,
            trace_bundle_dir, and error classification on failure. On any
            uncaught exception the evaluator must convert to a penalized
            EvalResult — never propagate.
        """
        ...


def cache_key(
    candidate: Candidate,
    budget: Budget,
    seed: int,
    dataset_version: str,
    evaluator_version: str,
) -> str:
    """Deterministic cache key for an evaluation (from parameter-golf).

    Uses budget.stable_hash() so any budget change — including a changed
    env_override — automatically invalidates the cache for that stage.
    Identical key → reuse stored EvalResult.
    """
    from hashlib import blake2b

    parts = [
        candidate.text_hash(),
        candidate.prog_hash(),
        budget.stable_hash(),
        f"seed{seed}",
        dataset_version,
        evaluator_version,
    ]
    return blake2b("|".join(parts).encode(), digest_size=12).hexdigest()


def materialize_prog_genome(
    candidate: Candidate,
    budget: Budget,
    *,
    base_prog: dict[str, Any] | None = None,
    patch_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce the effective prog-config a subprocess should receive.

    Merge order (later wins):
      base_prog          # registry seed values / defaults
      candidate.prog_genome  # optimizer's proposal
      patch_env          # resolved patch template env_overrides
      budget.env_overrides   # stage-specific overrides (e.g. ITERATIONS=50)

    The result is a flat dict of str → Any that callers turn into env vars
    or YAML overrides depending on their runner.

    From parameter-golf's CanonicalEvaluator.materialize_prog_genome().
    """
    result: dict[str, Any] = {}
    if base_prog:
        result.update(base_prog)
    result.update(candidate.prog_genome)
    if patch_env:
        result.update(patch_env)
    result.update(budget.env_overrides)
    return result


def load_eval_cache(
    cache_dir: Path,
    key: str,
) -> EvalResult | None:
    """Load a cached EvalResult, or return None on miss."""
    p = cache_dir / f"{key}.json"
    if not p.exists():
        return None
    try:
        return EvalResult.from_dict(json.loads(p.read_text()))
    except Exception:
        return None


def store_eval_cache(cache_dir: Path, key: str, result: EvalResult) -> None:
    """Persist an EvalResult to the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str)
    )
