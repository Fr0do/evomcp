"""EvoX-style evolutionary runner.

Evolves `candidate.prog_genome` against a fixed `candidate.text_genome`.

Concrete implementation based on parameter-golf's evox_runner.py:
  - sample_prog_genome() / mutate_prog_genome() delegated to registry
  - Seen-hash deduplication (no re-evaluating identical candidates)
  - Pareto archive maintained by dominance across objectives
  - State serialization (state.json) for --resume support
  - events.jsonl per run (one JSON per candidate × generation)
  - pareto_archive.json written after every generation

Algorithm:
  1. Init population of size N with random prog_genomes
  2. For each generation:
     a. Evaluate each member (possibly across multiple seeds)
     b. Score multi-seed mean primary + secondary objectives
     c. Update Pareto archive
     d. Elite selection: keep top floor(N/2)
     e. Mutate elites to fill next generation (plus random immigrants)
     f. Checkpoint state.json + pareto_archive.json

Multi-objective backend (Step 5): swap in EvoX JAX/NSGA-II for population
selection. The evaluator interface here is already wired; only the selection
step needs upgrading.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from evomcp.pipeline.candidate import Budget, Candidate, EvalResult, FAILURE_PENALTY
from evomcp.pipeline.evaluator import (
    Evaluator,
    cache_key,
    load_eval_cache,
    store_eval_cache,
    materialize_prog_genome,
)
from evomcp.pipeline.metrics import pareto_front
from evomcp.pipeline.program_db import ProgramDB
from evomcp.pipeline.registry import DEFAULT_REGISTRY

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PopulationMember:
    prog_genome: dict[str, Any]
    parent_ids: tuple[str, ...] = ()
    source: str = "random"


# ---------------------------------------------------------------------------
# State (checkpoint / resume)
# ---------------------------------------------------------------------------

@dataclass
class EvoXState:
    """Serializable runner state for --resume.

    From parameter-golf: saves generation counter + RNG state so a resumed
    run continues exactly where it left off (same random sequence).
    """

    generation: int = 0
    rng_state: list = field(default_factory=list)   # random.Random().getstate()
    seen_hashes: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "rng_state": list(self.rng_state),
            "seen_hashes": sorted(self.seen_hashes),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvoXState":
        return cls(
            generation=int(d.get("generation", 0)),
            rng_state=list(d.get("rng_state", [])),
            seen_hashes=set(d.get("seen_hashes", [])),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "EvoXState":
        return cls.from_dict(json.loads(path.read_text()))


# ---------------------------------------------------------------------------
# Pareto archive
# ---------------------------------------------------------------------------

@dataclass
class ParetoArchive:
    """Program-candidate Pareto archive.

    Stores (candidate, mean_result) pairs; updates by dominance.
    Objectives are (primary_score ↑, latency_s ↓, failure_rate ↓).
    From parameter-golf's dominates() + archive update pattern.
    """

    entries: list[tuple[Candidate, EvalResult]] = field(default_factory=list)
    objectives: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("primary_score", "max"),
            ("latency_s",     "min"),
        ]
    )

    def update(self, candidate: Candidate, result: EvalResult) -> bool:
        """Add (candidate, result) if non-dominated. Returns True if added."""
        if not result.success:
            return False
        # Remove dominated entries
        self.entries = [
            (c, r) for c, r in self.entries
            if not _dominates(result, r, self.objectives)
        ]
        # Check if the new entry is dominated by any existing
        for _, existing in self.entries:
            if _dominates(existing, result, self.objectives):
                return False
        self.entries.append((candidate, result))
        return True

    def best(self) -> tuple[Candidate, EvalResult] | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda x: x[1].primary_score)

    def to_list(self) -> list[dict]:
        out = []
        for c, r in self.entries:
            out.append({"candidate": c.to_dict(), "result": r.to_dict()})
        return out

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_list(), indent=2, default=str))


def _dominates(
    a: EvalResult,
    b: EvalResult,
    objectives: list[tuple[str, str]],
) -> bool:
    """Return True iff a Pareto-dominates b under the given objectives."""
    def get(r: EvalResult, name: str) -> float:
        return r.primary_score if name == "primary_score" else r.secondary_scores.get(name, 0.0)

    better_any = False
    for name, direction in objectives:
        av, bv = get(a, name), get(b, name)
        if direction == "min":
            av, bv = -av, -bv
        if av < bv:
            return False    # a is worse on this objective
        if av > bv:
            better_any = True
    return better_any


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _mean_result(results: list[EvalResult], fixed_text: dict, prog: dict) -> EvalResult:
    """Aggregate multi-seed results into one representative EvalResult."""
    successes = [r for r in results if r.success]
    if not successes:
        return results[0] if results else EvalResult.penalized(
            "unknown", __import__("pipeline.candidate", fromlist=["FailureClass"]).FailureClass.RUNTIME, "no results"
        )
    primary = sum(r.primary_score for r in successes) / len(successes)
    # Aggregate secondary scores
    keys = set().union(*(r.secondary_scores for r in successes))
    secondary = {k: sum(r.secondary_scores.get(k, 0) for r in successes) / len(successes) for k in keys}
    best = max(successes, key=lambda r: r.primary_score)
    from evomcp.pipeline.candidate import CostMetrics
    total_cost = CostMetrics(
        usd=sum(r.cost.usd for r in results),
        wall_s=sum(r.cost.wall_s for r in results),
        calls=sum(r.cost.calls for r in results),
        input_tokens=sum(r.cost.input_tokens for r in results),
        output_tokens=sum(r.cost.output_tokens for r in results),
    )
    return EvalResult(
        candidate_id=best.candidate_id,
        success=True,
        primary_score=primary,
        secondary_scores=secondary,
        cost=total_cost,
        trace_bundle_dir=best.trace_bundle_dir,
        evaluator_version=best.evaluator_version,
        seed=best.seed,
        dataset_version=best.dataset_version,
        stage=best.stage,
    )


def _append_event(path: Path, event: dict) -> None:
    """Append one JSON event to a run-level events.jsonl."""
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(event, default=str) + "\n")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def run(
    config_path: str | Path,
    evaluator: Evaluator | None = None,
    *,
    resume: bool = False,
) -> ParetoArchive:
    """EvoX-only run.

    Args:
        config_path: path to configs/evox.yaml (or compatible).
        evaluator: concrete Evaluator instance. If None, the run validates
            config + search space only (dry-run mode).
        resume: if True, load state.json from output_dir and continue.

    Returns:
        ParetoArchive of non-dominated candidates found during the run.
    """
    cfg = load_config(config_path)
    evox_cfg = cfg.get("evox", {})

    population_size: int = evox_cfg.get("population_size", 8)
    generations: int = evox_cfg.get("generations", 10)
    target_slots: list[str] = cfg.get("target_prog_slots", list(DEFAULT_REGISTRY.prog_slots))
    fixed_text: dict = cfg.get("fixed_text_genome", {})
    eval_seeds: list[int] = evox_cfg.get("evaluation_seeds", [0])
    elite_fraction: float = evox_cfg.get("elite_fraction", 0.5)
    immigrant_fraction: float = evox_cfg.get("immigrant_fraction", 0.25)
    stage_gates: dict = evox_cfg.get("stage_gates", {})
    db_selection_cfg = _load_db_selection_cfg(cfg, evox_cfg)

    output_dir = Path(cfg.get("output_dir", "artifacts/runs/evox"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.get("cache", {}).get("dir", "artifacts/cache"))
    events_path = output_dir / "events.jsonl"
    state_path = output_dir / "state.json"
    archive_path = output_dir / "pareto_archive.json"
    program_db = ProgramDB.from_config(
        cfg,
        config_path=config_path,
        output_dir=output_dir,
        mode="evox",
    )

    # Validate
    text_errs = DEFAULT_REGISTRY.validate_text_genome(fixed_text)
    if text_errs:
        raise ValueError(f"invalid fixed_text_genome: {text_errs}")

    # Select budget
    budgets = _load_budgets(cfg)
    budget = budgets.get("proxy", budgets.get("stage1", list(budgets.values())[0]))

    # Resume or init
    rng = random.Random()
    state = EvoXState()
    if resume and state_path.exists():
        state = EvoXState.load(state_path)
        if state.rng_state:
            rng.setstate(tuple(state.rng_state))  # type: ignore
        log.info("Resumed from generation %d (%d seen)", state.generation, len(state.seen_hashes))

    archive = ParetoArchive()

    # Build initial population
    population: list[PopulationMember] = _initial_population(
        population_size=population_size,
        target_slots=target_slots,
        rng=rng,
        program_db=program_db,
        db_selection_cfg=db_selection_cfg,
    )

    log.info(
        "EvoX: pop=%d gens=%d slots=%s db_selection=%s",
        population_size,
        generations,
        target_slots,
        "on" if db_selection_cfg["enabled"] and program_db else "off",
    )

    for gen in range(state.generation, generations):
        t0 = time.monotonic()
        gen_results: list[tuple[Candidate, EvalResult]] = []

        for member in population:
            prog_genome = member.prog_genome
            candidate = Candidate(
                text_genome=fixed_text,
                prog_genome=prog_genome,
                parent_ids=member.parent_ids,
                metadata={"mode": "evox", "generation": gen, "source": member.source},
            )
            cid = candidate.candidate_id
            if program_db:
                program_db.record_program(candidate, generation=gen)

            # Dedup
            if cid in state.seen_hashes:
                log.debug("skip seen %s", cid[:8])
                continue
            state.seen_hashes.add(cid)

            # Multi-seed evaluation
            seed_results: list[EvalResult] = []
            for seed in eval_seeds:
                ck = cache_key(
                    candidate, budget, seed,
                    cfg.get("evaluator", {}).get("dataset_version", ""),
                    cfg.get("evaluator", {}).get("version", ""),
                )
                cached = load_eval_cache(cache_dir, ck)
                if cached is not None:
                    seed_results.append(cached)
                    if program_db:
                        program_db.record_evaluation(
                            candidate,
                            cached,
                            generation=gen,
                            seed=seed,
                            budget=budget,
                        )
                    continue
                if evaluator is None:
                    log.warning("dry-run: no evaluator provided, skipping eval")
                    break
                result = evaluator.evaluate(
                    candidate, budget, seed, run_dir=output_dir / "traces"
                )
                store_eval_cache(cache_dir, ck, result)
                seed_results.append(result)
                if program_db:
                    program_db.record_evaluation(
                        candidate,
                        result,
                        generation=gen,
                        seed=seed,
                        budget=budget,
                    )

            if not seed_results:
                continue
            mean = _mean_result(seed_results, fixed_text, prog_genome)
            gen_results.append((candidate, mean))
            archive.update(candidate, mean)
            event = {
                "type": "candidate",
                "generation": gen,
                "candidate_id": cid[:12],
                "primary_score": mean.primary_score,
                "success": mean.success,
                "secondary_scores": mean.secondary_scores,
            }
            _append_event(events_path, event)
            if program_db:
                program_db.record_evaluation(
                    candidate,
                    mean,
                    generation=gen,
                    seed=-1,
                    budget=budget,
                    is_aggregate=True,
                )
                program_db.record_event(event)

        # --- selection ---------------------------------------------------
        successes = [(c, r) for c, r in gen_results if r.success]
        successes.sort(key=lambda x: x[1].primary_score, reverse=True)
        n_elite = max(1, int(population_size * elite_fraction))
        elites = successes[:n_elite]

        log.info(
            "gen %d/%d | pop=%d success=%d | best=%.4f | wall=%.1fs",
            gen + 1, generations,
            len(gen_results), len(successes),
            elites[0][1].primary_score if elites else float("-inf"),
            time.monotonic() - t0,
        )

        # Stage gate: prune at proxy stage if too few successes
        proxy_threshold = stage_gates.get("stage1_threshold")
        if proxy_threshold and elites and elites[0][1].primary_score < proxy_threshold:
            log.warning("All elites below proxy threshold %.3f; widening search.", proxy_threshold)

        # Build next population
        next_population: list[PopulationMember] = []
        n_immigrants = max(1, int(population_size * immigrant_fraction))
        for c, _ in elites:
            next_population.append(_mutated_member(c, target_slots, rng, "elite_mutation"))
        db_parent_pool = _db_parent_pool(program_db, target_slots, db_selection_cfg)
        n_db_parents = min(
            len(db_parent_pool),
            max(0, int(population_size * db_selection_cfg["parent_fraction"])),
        )
        for parent in rng.sample(db_parent_pool, n_db_parents) if n_db_parents else []:
            next_population.append(_mutated_db_member(parent, target_slots, rng))
        for _ in range(n_immigrants):
            next_population.append(_random_member(target_slots, rng))
        while len(next_population) < population_size:
            parent = rng.choice(elites)[0] if elites else None
            if parent:
                next_population.append(_mutated_member(parent, target_slots, rng, "elite_mutation"))
            else:
                db_parent = rng.choice(db_parent_pool) if db_parent_pool else None
                if db_parent:
                    next_population.append(_mutated_db_member(db_parent, target_slots, rng))
                else:
                    next_population.append(_random_member(target_slots, rng))
        population = next_population[:population_size]

        # Checkpoint
        state.generation = gen + 1
        state.rng_state = list(rng.getstate())
        state.save(state_path)
        archive.save(archive_path)
        if program_db:
            program_db.record_archive(archive.entries, generation=gen)

        summary_event = {
            "type": "generation_summary",
            "generation": gen,
            "n_evaluated": len(gen_results),
            "n_success": len(successes),
            "archive_size": len(archive.entries),
            "best_primary": elites[0][1].primary_score if elites else None,
        }
        _append_event(events_path, summary_event)
        if program_db:
            program_db.record_event(summary_event)

    # Final summary
    best = archive.best()
    if best:
        cand, res = best
        log.info(
            "EvoX complete. Best: %s primary=%.4f %s",
            cand.candidate_id[:12],
            res.primary_score,
            res.secondary_scores,
        )
    if program_db:
        program_db.record_run(status="complete")
        program_db.close()

    return archive


# ---------------------------------------------------------------------------
# Budget loader (shared with other runners)
# ---------------------------------------------------------------------------

def _load_budgets(cfg: dict) -> dict[str, Budget]:
    """Parse the budgets list from base config into a name → Budget dict."""
    out = {}
    for bd in cfg.get("budgets", []):
        b = Budget(
            stage=int(bd.get("stage", 0)),
            max_plans=int(bd.get("max_plans", 0)),
            max_judges=int(bd.get("max_judges", 0)),
            timeout_s=int(bd.get("timeout_s", 60)),
            env_overrides=dict(bd.get("env_overrides", {})),
        )
        out[bd.get("name", f"stage{b.stage}")] = b
    return out


def _load_db_selection_cfg(cfg: dict[str, Any], evox_cfg: dict[str, Any]) -> dict[str, Any]:
    db_cfg = cfg.get("program_db", cfg.get("database", {}))
    selection_cfg = dict(db_cfg.get("selection", {}) if isinstance(db_cfg, dict) else {})
    selection_cfg.update(evox_cfg.get("program_db_selection", {}))
    return {
        "enabled": bool(selection_cfg.get("enabled", True)),
        "archive_name": str(selection_cfg.get("archive_name", "pareto")),
        "scope": str(selection_cfg.get("scope", "all_runs")),
        "seed_fraction": float(selection_cfg.get("seed_fraction", 0.25)),
        "parent_fraction": float(selection_cfg.get("parent_fraction", 0.25)),
        "limit": int(selection_cfg.get("limit", 64)),
        "min_score": selection_cfg.get("min_score"),
    }


def _initial_population(
    *,
    population_size: int,
    target_slots: list[str],
    rng: random.Random,
    program_db: ProgramDB | None,
    db_selection_cfg: dict[str, Any],
) -> list[PopulationMember]:
    population: list[PopulationMember] = []
    db_pool = _db_parent_pool(program_db, target_slots, db_selection_cfg)
    n_seed = min(
        len(db_pool),
        max(0, int(population_size * db_selection_cfg["seed_fraction"])),
    )
    for parent in rng.sample(db_pool, n_seed) if n_seed else []:
        _append_unique_member(
            population,
            PopulationMember(
                prog_genome=dict(parent["prog_genome"]),
                parent_ids=(str(parent["candidate_id"]),),
                source="db_archive_seed",
            ),
        )
    attempts = 0
    while len(population) < population_size:
        if _append_unique_member(population, _random_member(target_slots, rng)):
            attempts = 0
            continue
        attempts += 1
        if attempts > population_size * 10:
            population.append(_random_member(target_slots, rng))
            attempts = 0
    return population


def _db_parent_pool(
    program_db: ProgramDB | None,
    target_slots: list[str],
    db_selection_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if program_db is None or not db_selection_cfg["enabled"]:
        return []
    parents = program_db.parent_programs(
        limit=int(db_selection_cfg["limit"]),
        archive_name=str(db_selection_cfg["archive_name"]),
        scope=str(db_selection_cfg["scope"]),
        min_score=db_selection_cfg["min_score"],
    )
    valid: list[dict[str, Any]] = []
    for parent in parents:
        genome = parent.get("prog_genome") or {}
        if not isinstance(genome, dict):
            continue
        if any(slot not in genome for slot in target_slots):
            continue
        filtered = {slot: genome[slot] for slot in target_slots}
        if DEFAULT_REGISTRY.validate_prog_genome(filtered):
            continue
        item = dict(parent)
        item["prog_genome"] = filtered
        valid.append(item)
    return valid


def _random_member(target_slots: list[str], rng: random.Random) -> PopulationMember:
    return PopulationMember(
        prog_genome=DEFAULT_REGISTRY.sample_prog_genome(target_slots, rng),
        source="random_immigrant",
    )


def _mutated_member(
    parent: Candidate,
    target_slots: list[str],
    rng: random.Random,
    source: str,
) -> PopulationMember:
    return PopulationMember(
        prog_genome=DEFAULT_REGISTRY.mutate_prog_genome(dict(parent.prog_genome), target_slots, rng),
        parent_ids=(parent.candidate_id,),
        source=source,
    )


def _mutated_db_member(
    parent: dict[str, Any],
    target_slots: list[str],
    rng: random.Random,
) -> PopulationMember:
    return PopulationMember(
        prog_genome=DEFAULT_REGISTRY.mutate_prog_genome(dict(parent["prog_genome"]), target_slots, rng),
        parent_ids=(str(parent["candidate_id"]),),
        source="db_archive_mutation",
    )


def _append_unique_member(population: list[PopulationMember], member: PopulationMember) -> bool:
    signature = json.dumps(member.prog_genome, sort_keys=True, default=str)
    for existing in population:
        if json.dumps(existing.prog_genome, sort_keys=True, default=str) == signature:
            return False
    population.append(member)
    return True


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(sys.argv[1])
