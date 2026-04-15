"""Hybrid runner: outer EvoX (prog), inner GEPA (text).

Never invert this: program search outside, prompt search inside.
Prompts evolved against a broken program produce noise; programs evolved
against a broken prompt discard correct programs.

Concrete implementation from parameter-golf's hybrid_runner.py:

  For each prog_genome in population:
    1. Evaluate raw candidate (fixed text, proposed prog) → raw_result
    2. IF raw_score >= gepa_trigger_score:
         Run one GEPA inner step → refined text_genome
         Evaluate refined candidate → refined_result
    3. Log both raw + refined in a single event
    4. Next generation: selection/mutation on prog surface only

  gepa_trigger_score defaults to -inf (= always run GEPA), but in practice
  we set it to the stage-1 proxy floor so GEPA isn't wasted on bad programs.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evomcp.pipeline.candidate import Budget, Candidate, EvalResult
from evomcp.pipeline.evaluator import Evaluator, cache_key, load_eval_cache, store_eval_cache
from evomcp.pipeline.registry import DEFAULT_REGISTRY
from evomcp.optim.evox_runner import (
    EvoXState,
    ParetoArchive,
    _append_event,
    _dominates,
    _load_budgets,
)
from evomcp.optim.gepa_runner import TextParetoArchive

log = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def run(
    config_path: str | Path,
    evaluator: Evaluator | None = None,
    *,
    resume: bool = False,
) -> tuple[ParetoArchive, TextParetoArchive]:
    """Hybrid run: outer EvoX + inner GEPA.

    Returns:
        (ParetoArchive of prog candidates, TextParetoArchive of text candidates)
    """
    cfg = load_config(config_path)
    evox_cfg = cfg.get("evox", {})
    inner_gepa_cfg = cfg.get("inner_gepa", {})
    trigger_cfg = cfg.get("inner_gepa_trigger", {})

    population_size: int = evox_cfg.get("population_size", 8)
    generations: int = evox_cfg.get("generations", 10)
    prog_slots: list[str] = cfg.get("target_prog_slots", list(DEFAULT_REGISTRY.prog_slots))
    text_slots: list[str] = cfg.get("target_text_slots", list(DEFAULT_REGISTRY.text_slots))
    initial_text: dict = cfg.get("initial_text_genome", {})
    eval_seeds: list[int] = evox_cfg.get("evaluation_seeds", [0])
    elite_fraction: float = evox_cfg.get("elite_fraction", 0.5)
    gepa_trigger_score: float = float(trigger_cfg.get("min_proxy_score", float("-inf")))
    inner_rounds: int = inner_gepa_cfg.get("generations", inner_gepa_cfg.get("rounds", 3))
    inner_metric_calls: int = inner_gepa_cfg.get("max_metric_calls", inner_gepa_cfg.get("population_size", 4) * 2)

    output_dir = Path(cfg.get("output_dir", "artifacts/runs/hybrid"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.get("cache", {}).get("dir", "artifacts/cache"))
    events_path = output_dir / "events.jsonl"
    state_path = output_dir / "state.json"
    prog_archive_path = output_dir / "pareto_archive.json"
    text_archive_path = output_dir / "text_pareto_archive.json"

    # Budgets
    budgets = _load_budgets(cfg)
    proxy_budget = budgets.get("proxy", budgets.get("stage1", list(budgets.values())[0] if budgets else Budget(
        stage=1, max_plans=10, max_judges=1, timeout_s=600
    )))

    # Initial text genome from registry seeds if not provided
    seed_text: dict[str, str] = {
        name: DEFAULT_REGISTRY.text_slots[name].seed_value
        for name in text_slots
        if name in DEFAULT_REGISTRY.text_slots and DEFAULT_REGISTRY.text_slots[name].mutatable
    }
    if initial_text:
        seed_text.update(initial_text)

    # Resume / init
    rng = random.Random()
    state = EvoXState()
    if resume and state_path.exists():
        state = EvoXState.load(state_path)
        if state.rng_state:
            rng.setstate(tuple(state.rng_state))  # type: ignore
        log.info("Resumed from generation %d", state.generation)

    prog_archive = ParetoArchive()
    text_archive = TextParetoArchive()

    # Initial population
    population: list[dict[str, Any]] = [
        DEFAULT_REGISTRY.sample_prog_genome(prog_slots, rng)
        for _ in range(population_size)
    ]

    # Current shared text genome (inner GEPA refines this per-candidate)
    current_text_genome = dict(seed_text)

    log.info(
        "Hybrid: pop=%d gens=%d prog_slots=%s text_slots=%s trigger=%.3f",
        population_size, generations, prog_slots, text_slots, gepa_trigger_score,
    )

    for gen in range(state.generation, generations):
        t0 = time.monotonic()
        gen_events: list[dict] = []

        for prog_genome in population:
            cid_raw = Candidate(
                text_genome=current_text_genome,
                prog_genome=prog_genome,
                metadata={"mode": "hybrid_raw", "generation": gen},
            ).candidate_id

            if cid_raw in state.seen_hashes:
                log.debug("skip seen %s", cid_raw[:8])
                continue

            # --- Step 1: Raw eval (fixed text, proposed prog) ---------------
            raw_candidate = Candidate(
                text_genome=current_text_genome,
                prog_genome=prog_genome,
                metadata={"mode": "hybrid_raw", "generation": gen},
            )
            raw_results = _eval_multi_seed(
                raw_candidate, proxy_budget, eval_seeds, evaluator,
                cache_dir, cfg, output_dir
            )
            state.seen_hashes.add(cid_raw)

            raw_success = [r for r in raw_results if r.success]
            raw_mean_score = (
                sum(r.primary_score for r in raw_success) / len(raw_success)
                if raw_success else float("-inf")
            )
            raw_mean = _make_mean_result(raw_results, raw_candidate.candidate_id)
            prog_archive.update(raw_candidate, raw_mean)

            # --- Step 2: Conditional inner GEPA ----------------------------
            refined_mean: EvalResult | None = None
            refined_text: dict | None = None

            if raw_mean_score >= gepa_trigger_score:
                log.debug(
                    "gen %d: raw=%.4f >= trigger=%.4f → inner GEPA",
                    gen, raw_mean_score, gepa_trigger_score,
                )
                refined_text, refined_mean = _run_inner_gepa(
                    current_text_genome=current_text_genome,
                    fixed_prog=prog_genome,
                    text_slots=text_slots,
                    budget=proxy_budget,
                    eval_seeds=eval_seeds,
                    evaluator=evaluator,
                    cache_dir=cache_dir,
                    cfg=cfg,
                    output_dir=output_dir,
                    rounds=inner_rounds,
                    inner_gepa_cfg=inner_gepa_cfg,
                )
                if refined_text and refined_mean and refined_mean.success:
                    refined_candidate = Candidate(
                        text_genome=refined_text,
                        prog_genome=prog_genome,
                        metadata={"mode": "hybrid_refined", "generation": gen},
                    )
                    text_archive.update(refined_candidate, refined_mean)
                    # Use refined text as new baseline if it's better
                    if refined_mean.primary_score > raw_mean_score:
                        current_text_genome = dict(refined_text)
                        log.debug(
                            "Updated shared text genome: %.4f → %.4f",
                            raw_mean_score, refined_mean.primary_score,
                        )
            else:
                log.debug(
                    "gen %d: raw=%.4f < trigger=%.4f → skip GEPA",
                    gen, raw_mean_score, gepa_trigger_score,
                )

            gen_events.append({
                "type": "hybrid_candidate",
                "generation": gen,
                "prog_id": cid_raw[:12],
                "raw_score": raw_mean_score,
                "refined_score": refined_mean.primary_score if refined_mean else None,
                "gepa_triggered": refined_text is not None,
                "n_raw_success": len(raw_success),
            })

        # --- Selection + next population -----------------------------------
        all_results = prog_archive.entries
        successes = [(c, r) for c, r in all_results if r.success]
        successes.sort(key=lambda x: x[1].primary_score, reverse=True)
        n_elite = max(1, int(population_size * elite_fraction))
        elites = successes[:n_elite]

        log.info(
            "gen %d/%d | pop=%d | archive=%d | best=%.4f | wall=%.1fs",
            gen + 1, generations,
            len(population), len(prog_archive.entries),
            elites[0][1].primary_score if elites else float("-inf"),
            time.monotonic() - t0,
        )

        next_population = []
        for c, _ in elites:
            next_population.append(
                DEFAULT_REGISTRY.mutate_prog_genome(dict(c.prog_genome), prog_slots, rng)
            )
        while len(next_population) < population_size:
            if elites:
                parent = rng.choice(elites)[0]
                next_population.append(
                    DEFAULT_REGISTRY.mutate_prog_genome(dict(parent.prog_genome), prog_slots, rng)
                )
            else:
                next_population.append(DEFAULT_REGISTRY.sample_prog_genome(prog_slots, rng))
        population = next_population[:population_size]

        # Checkpoint
        state.generation = gen + 1
        state.rng_state = list(rng.getstate())
        state.save(state_path)
        prog_archive.save(prog_archive_path)
        text_archive.save(text_archive_path)

        for ev in gen_events:
            _append_event(events_path, ev)
        _append_event(events_path, {
            "type": "generation_summary",
            "generation": gen,
            "prog_archive_size": len(prog_archive.entries),
            "text_archive_size": len(text_archive.entries),
            "best_primary": elites[0][1].primary_score if elites else None,
        })

    best_prog = prog_archive.best()
    if best_prog:
        log.info("Hybrid complete. Best prog: %s score=%.4f",
                 best_prog[0].candidate_id[:12], best_prog[1].primary_score)
    best_text = text_archive.best()
    if best_text:
        log.info("Hybrid complete. Best text: %s score=%.4f",
                 best_text[0].candidate_id[:12], best_text[1].primary_score)

    return prog_archive, text_archive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_multi_seed(
    candidate: Candidate,
    budget: Budget,
    seeds: list[int],
    evaluator: Evaluator | None,
    cache_dir: Path,
    cfg: dict,
    run_dir: Path,
) -> list[EvalResult]:
    results = []
    for seed in seeds:
        ck = cache_key(
            candidate, budget, seed,
            cfg.get("evaluator", {}).get("dataset_version", ""),
            cfg.get("evaluator", {}).get("version", ""),
        )
        cached = load_eval_cache(cache_dir, ck)
        if cached is not None:
            results.append(cached)
            continue
        if evaluator is None:
            continue
        result = evaluator.evaluate(candidate, budget, seed, run_dir=run_dir / "traces")
        store_eval_cache(cache_dir, ck, result)
        results.append(result)
    return results


def _make_mean_result(results: list[EvalResult], candidate_id: str) -> EvalResult:
    from evomcp.pipeline.candidate import CostMetrics, FailureClass
    successes = [r for r in results if r.success]
    if not successes:
        err = results[0].error_type if results else FailureClass.RUNTIME
        msg = results[0].error_message if results else "no results"
        return EvalResult.penalized(candidate_id, err or FailureClass.RUNTIME, msg or "")
    primary = sum(r.primary_score for r in successes) / len(successes)
    keys = set().union(*(r.secondary_scores for r in successes))
    secondary = {k: sum(r.secondary_scores.get(k, 0) for r in successes) / len(successes) for k in keys}
    best = max(successes, key=lambda r: r.primary_score)
    total_cost = CostMetrics(
        usd=sum(r.cost.usd for r in results),
        wall_s=sum(r.cost.wall_s for r in results),
        calls=sum(r.cost.calls for r in results),
        input_tokens=sum(r.cost.input_tokens for r in results),
        output_tokens=sum(r.cost.output_tokens for r in results),
    )
    return EvalResult(
        candidate_id=candidate_id,
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


def _run_inner_gepa(
    current_text_genome: dict,
    fixed_prog: dict,
    text_slots: list[str],
    budget: Budget,
    eval_seeds: list[int],
    evaluator: Evaluator | None,
    cache_dir: Path,
    cfg: dict,
    output_dir: Path,
    rounds: int,
    inner_gepa_cfg: dict,
) -> tuple[dict | None, EvalResult | None]:
    """Run a short GEPA refinement step inline (no subprocess).

    This is the inner GEPA from parameter-golf's hybrid_runner. It mirrors
    optim/gepa_runner.run() but is called as a function rather than a separate
    process, sharing the parent's evaluator and cache.

    Returns (refined_text_genome, mean_EvalResult) or (None, None) on failure.
    """
    try:
        import dspy

        mutation_backend: dict = inner_gepa_cfg.get("mutation_backend", {})
        lm_model = mutation_backend.get("model", "claude-haiku-4-5")
        lm_backend = mutation_backend.get("backend", "claude")

        if lm_backend == "claude":
            lm = dspy.LM(f"anthropic/{lm_model}", max_tokens=4096)
        else:
            lm = dspy.LM(lm_model, max_tokens=4096)
        dspy.configure(lm=lm)

        from evomcp.optim.gepa_runner import _build_signatures
        PromptMutationSignature, PlannerSignature = _build_signatures()
        mutator = dspy.ChainOfThought(PromptMutationSignature)
        planner = dspy.ChainOfThought(PlannerSignature)

        best_text = dict(current_text_genome)
        best_score: float = float("-inf")
        best_result: EvalResult | None = None

        for r in range(rounds):
            candidate = Candidate(
                text_genome=best_text,
                prog_genome=fixed_prog,
                metadata={"mode": "hybrid_inner_gepa", "round": r},
            )
            seed_results = _eval_multi_seed(
                candidate, budget, eval_seeds, evaluator, cache_dir, cfg, output_dir
            )
            successes = [x for x in seed_results if x.success]
            if successes:
                mean_score = sum(x.primary_score for x in successes) / len(successes)
                if mean_score > best_score:
                    best_score = mean_score
                    best_result = max(successes, key=lambda x: x.primary_score)

            if r == rounds - 1:
                break

            # Mutate
            from evomcp.pipeline.tracing import trace_feedback_summary
            feedback = []
            for sr in seed_results:
                if sr.trace_bundle_dir and Path(str(sr.trace_bundle_dir)).exists():
                    feedback.append(trace_feedback_summary(sr.trace_bundle_dir))

            constraints_dict = {
                name: {
                    "max_chars": DEFAULT_REGISTRY.text_slots[name].max_chars,
                    "role": DEFAULT_REGISTRY.text_slots[name].role,
                }
                for name in text_slots
                if name in DEFAULT_REGISTRY.text_slots
            }
            mut_out = mutator(
                slot_names=", ".join(text_slots),
                current_texts=json.dumps(best_text, ensure_ascii=False),
                trace_feedback=json.dumps(feedback, default=str),
            )
            proposed = json.loads(mut_out.revised_texts)
            plan_out = planner(
                slot_names=", ".join(text_slots),
                revised_texts=json.dumps(proposed, ensure_ascii=False),
                constraints=json.dumps(constraints_dict, ensure_ascii=False),
            )
            accepted = json.loads(plan_out.accepted_texts)
            errs = DEFAULT_REGISTRY.validate_text_genome(accepted)
            if not errs:
                best_text = {**best_text, **{k: v for k, v in accepted.items() if k in text_slots}}

        return (best_text, best_result) if best_result else (None, None)

    except Exception as exc:
        log.warning("inner GEPA failed: %s", exc)
        return None, None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(sys.argv[1])
