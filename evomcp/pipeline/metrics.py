"""Metric aggregation.

Supports: weighted scalar, Pareto fronts, hard constraints. Never collapse
to a single headline metric — the optimizer must see the full tradeoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from evomcp.pipeline.candidate import EvalResult


@dataclass(frozen=True)
class Constraint:
    name: str                   # secondary_scores key, or "primary_score"
    op: str                     # "<", "<=", ">", ">="
    threshold: float


def aggregate_weighted(
    result: EvalResult, weights: Mapping[str, float]
) -> float:
    """Weighted linear combination of primary + secondary scores."""
    total = 0.0
    for key, w in weights.items():
        if key == "primary_score":
            total += w * result.primary_score
        else:
            total += w * result.secondary_scores.get(key, 0.0)
    return total


def satisfies(result: EvalResult, constraints: Iterable[Constraint]) -> bool:
    for c in constraints:
        v = (
            result.primary_score
            if c.name == "primary_score"
            else result.secondary_scores.get(c.name, 0.0)
        )
        ok = {
            "<":  v <  c.threshold,
            "<=": v <= c.threshold,
            ">":  v >  c.threshold,
            ">=": v >= c.threshold,
        }[c.op]
        if not ok:
            return False
    return True


def pareto_front(
    results: Sequence[EvalResult],
    objectives: Sequence[tuple[str, str]],   # [(name, "max"|"min"), ...]
) -> list[EvalResult]:
    """Return non-dominated EvalResults for the given objectives.

    A candidate is dominated if some other candidate is >= on all objectives
    and strictly better on at least one.
    """
    def get(r: EvalResult, name: str) -> float:
        return r.primary_score if name == "primary_score" else r.secondary_scores.get(name, 0.0)

    def dominates(a: EvalResult, b: EvalResult) -> bool:
        better_any = False
        for name, direction in objectives:
            av, bv = get(a, name), get(b, name)
            if direction == "min":
                av, bv = -av, -bv
            if av < bv:
                return False
            if av > bv:
                better_any = True
        return better_any

    front = []
    for r in results:
        if not r.success:
            continue
        dominated = any(dominates(other, r) for other in results if other is not r and other.success)
        if not dominated:
            front.append(r)
    return front
