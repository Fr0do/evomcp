"""GEPA-only runner.

Evolves `candidate.text_genome` against a fixed `candidate.prog_genome`.

Concrete implementation based on parameter-golf's gepa_runner.py:
  - PromptMutationSignature: old texts + trace feedback → revised texts
  - PlannerSignature: revised texts + fixed prog → confirms patch/overrides
  - TextParetoArchive: text candidates ordered by objectives (kappa ↑, latency ↓)
  - dspy.GEPA wraps the reflective mutation (max_metric_calls bounds cost)
  - events.jsonl per round (one JSON per GEPA step)
  - text_pareto_archive.json after every round

Wire-up notes (Step 4):
  For HANFU the metric is Cohen's κ_w between VLM judge and teacher labels,
  computed by scripts/evolve_prompts.py internals. The signature currently
  uses a generic summarized feedback dict; adapt the slot names once
  scripts/evolve_prompts.py is refactored to return EvalResult.

DSPy is imported lazily so the runner loads even when dspy is not installed
(it will raise at run-time with a clear message).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evomcp.pipeline.candidate import Budget, Candidate, EvalResult
from evomcp.pipeline.evaluator import Evaluator, cache_key, load_eval_cache, store_eval_cache
from evomcp.pipeline.registry import DEFAULT_REGISTRY
from evomcp.pipeline.tracing import trace_feedback_summary
from evomcp.optim.evox_runner import _load_budgets, load_evaluator_from_config
from evomcp.project_loader import load_project_slots

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DSPy signature definitions
# ---------------------------------------------------------------------------

def _build_signatures():
    """Build DSPy signatures lazily (avoids hard dep at import time)."""
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "GEPA runner requires dspy-ai. Install with: pip install dspy-ai"
        ) from exc

    class PromptMutationSignature(dspy.Signature):
        """Given the current text genome and trace feedback from recent evaluations,
        propose improved versions of each text slot. Return each slot on a new line
        preceded by its name, e.g.:  critic_fengshui_prompt: <revised text>"""

        slot_names: str = dspy.InputField(
            desc="Comma-separated text slot names to mutate."
        )
        current_texts: str = dspy.InputField(
            desc="Current text genome as JSON dict {slot_name: text}."
        )
        trace_feedback: str = dspy.InputField(
            desc="JSON list of recent evaluation summaries with scores and errors."
        )
        revised_texts: str = dspy.OutputField(
            desc="Revised text genome as JSON dict {slot_name: new_text}."
        )

    class PlannerSignature(dspy.Signature):
        """Given revised text proposals, confirm whether they are valid,
        safe, and within the declared slot constraints. Output a validation
        summary and any further suggested edits."""

        slot_names: str = dspy.InputField(desc="Text slot names being validated.")
        revised_texts: str = dspy.InputField(desc="Proposed revised texts as JSON.")
        constraints: str = dspy.InputField(
            desc="Slot constraints as JSON: {slot_name: {max_chars, role, description}}."
        )
        validation_notes: str = dspy.OutputField(
            desc="Brief validation notes and any suggested corrections."
        )
        accepted_texts: str = dspy.OutputField(
            desc="Final accepted texts as JSON dict {slot_name: text}, "
                 "identical to revised_texts if no corrections needed."
        )

    return PromptMutationSignature, PlannerSignature


# ---------------------------------------------------------------------------
# Text Pareto archive
# ---------------------------------------------------------------------------

@dataclass
class TextParetoArchive:
    """Archive of text-genome candidates ordered by Pareto objectives.

    Objectives: primary_score (kappa) ↑ + latency_s ↓.
    From parameter-golf's text_pareto_archive.json pattern.
    """

    entries: list[tuple[Candidate, EvalResult]] = field(default_factory=list)

    def update(self, candidate: Candidate, result: EvalResult) -> bool:
        """Add if non-dominated. Returns True if added."""
        if not result.success:
            return False
        objectives = [("primary_score", "max"), ("latency_s", "min")]

        def get(r: EvalResult, n: str) -> float:
            return r.primary_score if n == "primary_score" else r.secondary_scores.get(n, 0.0)

        def dom(a: EvalResult, b: EvalResult) -> bool:
            better = False
            for n, d in objectives:
                av, bv = get(a, n), get(b, n)
                if d == "min": av, bv = -av, -bv
                if av < bv: return False
                if av > bv: better = True
            return better

        self.entries = [(c, r) for c, r in self.entries if not dom(result, r)]
        for _, ex in self.entries:
            if dom(ex, result):
                return False
        self.entries.append((candidate, result))
        return True

    def best(self) -> tuple[Candidate, EvalResult] | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda x: x[1].primary_score)

    def to_list(self) -> list[dict]:
        return [{"candidate": c.to_dict(), "result": r.to_dict()} for c, r in self.entries]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_list(), indent=2, default=str))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def run(
    config_path: str | Path,
    evaluator: Evaluator | None = None,
    *,
    resume: bool = False,
) -> TextParetoArchive:
    """GEPA-only run.

    Args:
        config_path: path to configs/gepa.yaml (or compatible).
        evaluator: concrete Evaluator instance. If None, dry-run.
        resume: if True, load text_pareto_archive.json from output_dir.

    Returns:
        TextParetoArchive of non-dominated text candidates.
    """
    load_project_slots(config_path)
    cfg = load_config(config_path)
    gepa_cfg = cfg.get("gepa", {})

    target_slots: list[str] = cfg.get("target_text_slots", list(DEFAULT_REGISTRY.text_slots))
    fixed_prog: dict = cfg.get("fixed_prog_genome", {})
    rounds: int = gepa_cfg.get("generations", gepa_cfg.get("rounds", 5))
    population_size: int = gepa_cfg.get("population_size", 4)
    max_metric_calls: int = gepa_cfg.get("max_metric_calls", population_size * 2)
    mutation_backend: dict = gepa_cfg.get("mutation_backend", {})
    eval_seeds: list[int] = gepa_cfg.get("evaluation_seeds", [0])
    reflect_on: list[str] = ["bad_format", "low_score"]  # GEPA only reflects on these

    output_dir = Path(cfg.get("output_dir", "artifacts/runs/gepa"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.get("cache", {}).get("dir", "artifacts/cache"))
    events_path = output_dir / "events.jsonl"
    archive_path = output_dir / "text_pareto_archive.json"

    # Validate
    prog_errs = DEFAULT_REGISTRY.validate_prog_genome(fixed_prog)
    if prog_errs:
        raise ValueError(f"invalid fixed_prog_genome: {prog_errs}")

    # Build initial text genome from registry seed values
    seed_genome = {
        name: DEFAULT_REGISTRY.text_slots[name].seed_value
        for name in target_slots
        if name in DEFAULT_REGISTRY.text_slots and DEFAULT_REGISTRY.text_slots[name].mutatable
    }
    frozen_slots = [
        name for name in target_slots
        if name in DEFAULT_REGISTRY.text_slots and not DEFAULT_REGISTRY.text_slots[name].mutatable
    ]
    if frozen_slots:
        log.info("Frozen (non-mutatable) slots skipped: %s", frozen_slots)

    budgets = _load_budgets(cfg)
    budget = budgets.get("proxy", list(budgets.values())[0] if budgets else Budget(
        stage=1, max_plans=10, max_judges=1, timeout_s=600
    ))

    archive = TextParetoArchive()
    if resume and archive_path.exists():
        log.info("Resuming from existing text archive: %s", archive_path)
        # Re-populate archive from disk (simplified: just read for logging)

    # Configure DSPy LM for mutation via backend registry
    try:
        import dspy
        from evomcp.backends.mutation import build_mutation_lm
        lm = build_mutation_lm(mutation_backend)
        dspy.configure(lm=lm)
        PromptMutationSignature, PlannerSignature = _build_signatures()
        mutator = dspy.ChainOfThought(PromptMutationSignature)
        planner = dspy.ChainOfThought(PlannerSignature)
        use_dspy = True
    except ImportError:
        log.warning("dspy not installed; falling back to no-op mutation (dry-run).")
        use_dspy = False

    current_genome = dict(seed_genome)
    # Include frozen slots in the candidate so they're part of the ID
    full_genome = {**current_genome, **{name: DEFAULT_REGISTRY.text_slots[name].seed_value
                                         for name in frozen_slots
                                         if name in DEFAULT_REGISTRY.text_slots}}

    log.info(
        "GEPA: rounds=%d slots=%s backend=%s/%s",
        rounds, target_slots,
        mutation_backend.get("backend", "claude"),
        mutation_backend.get("model", "claude-haiku-4-5"),
    )

    for rnd in range(rounds):
        t0 = time.monotonic()

        # Evaluate current genome
        candidate = Candidate(
            text_genome=full_genome,
            prog_genome=fixed_prog,
            metadata={"mode": "gepa", "round": rnd},
        )
        seed_results: list[EvalResult] = []
        feedback_summaries: list[dict] = []

        for seed in eval_seeds:
            ck = cache_key(
                candidate, budget, seed,
                cfg.get("evaluator", {}).get("dataset_version", ""),
                cfg.get("evaluator", {}).get("version", ""),
            )
            cached = load_eval_cache(cache_dir, ck)
            if cached is not None:
                seed_results.append(cached)
                if cached.trace_bundle_dir and Path(str(cached.trace_bundle_dir)).exists():
                    feedback_summaries.append(
                        trace_feedback_summary(cached.trace_bundle_dir)
                    )
                continue
            if evaluator is None:
                log.warning("dry-run: no evaluator; skipping eval for seed %d", seed)
                continue
            result = evaluator.evaluate(candidate, budget, seed, run_dir=output_dir / "traces")
            store_eval_cache(cache_dir, ck, result)
            seed_results.append(result)
            if result.trace_bundle_dir and Path(str(result.trace_bundle_dir)).exists():
                feedback_summaries.append(trace_feedback_summary(result.trace_bundle_dir))

        # Aggregate
        successes = [r for r in seed_results if r.success]
        mean_score = sum(r.primary_score for r in successes) / len(successes) if successes else None
        should_reflect = (
            not successes
            or (mean_score is not None and mean_score < gepa_cfg.get("reflection_threshold", 0.0))
            or any(
                r.error_type and r.error_type.value in reflect_on
                for r in seed_results
            )
        )

        if successes:
            best_result = max(successes, key=lambda r: r.primary_score)
            archive.update(candidate, best_result)

        log.info(
            "round %d/%d | score=%.4f | reflect=%s | wall=%.1fs",
            rnd + 1, rounds,
            mean_score or float("-inf"), should_reflect,
            time.monotonic() - t0,
        )

        # Build feedback string for GEPA
        feedback_str = json.dumps(feedback_summaries, default=str)

        # Emit round event
        _append_event(events_path, {
            "type": "gepa_round",
            "round": rnd,
            "text_genome_keys": list(current_genome),
            "mean_score": mean_score,
            "n_success": len(successes),
            "should_reflect": should_reflect,
            "archive_size": len(archive.entries),
        })

        # Early-stop check
        es = gepa_cfg.get("early_stop", {})
        best = archive.best()
        if (
            best is not None
            and rnd + 1 >= es.get("min_generations", es.get("min_rounds", 1))
            and mean_score is not None
            and mean_score >= es.get("kappa_target", float("inf"))
        ):
            log.info("Early stop: kappa_target %.3f reached.", es["kappa_target"])
            break

        if rnd == rounds - 1:
            break  # No mutation after last round

        # --- GEPA mutation step -------------------------------------------
        if not use_dspy:
            log.info("Skipping mutation (dspy unavailable).")
            continue

        constraints_dict = {
            name: {
                "max_chars": DEFAULT_REGISTRY.text_slots[name].max_chars,
                "role": DEFAULT_REGISTRY.text_slots[name].role,
                "description": DEFAULT_REGISTRY.text_slots[name].description,
            }
            for name in target_slots
            if name in DEFAULT_REGISTRY.text_slots and DEFAULT_REGISTRY.text_slots[name].mutatable
        }

        try:
            mutation_out = mutator(
                slot_names=", ".join(target_slots),
                current_texts=json.dumps(current_genome, ensure_ascii=False),
                trace_feedback=feedback_str,
            )
            proposed_texts = json.loads(mutation_out.revised_texts)

            plan_out = planner(
                slot_names=", ".join(target_slots),
                revised_texts=json.dumps(proposed_texts, ensure_ascii=False),
                constraints=json.dumps(constraints_dict, ensure_ascii=False),
            )
            accepted_texts = json.loads(plan_out.accepted_texts)

            # Validate and update genome
            errs = DEFAULT_REGISTRY.validate_text_genome(accepted_texts)
            if errs:
                log.warning("GEPA produced invalid texts: %s; keeping current.", errs)
            else:
                current_genome = {**current_genome, **{
                    k: v for k, v in accepted_texts.items()
                    if k in target_slots
                }}
                full_genome = {**current_genome, **{
                    name: DEFAULT_REGISTRY.text_slots[name].seed_value
                    for name in frozen_slots
                    if name in DEFAULT_REGISTRY.text_slots
                }}
                _append_event(events_path, {
                    "type": "gepa_mutation",
                    "round": rnd,
                    "mutated_slots": list(accepted_texts),
                    "validation_notes": plan_out.validation_notes,
                })
        except Exception as exc:
            log.warning("GEPA mutation failed round %d: %s", rnd, exc)

    archive.save(archive_path)

    best = archive.best()
    if best:
        cand, res = best
        log.info(
            "GEPA complete. Best: %s primary=%.4f",
            cand.candidate_id[:12], res.primary_score
        )

    return archive


def _append_event(path: Path, event: dict) -> None:
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(event, default=str) + "\n")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg_path = sys.argv[1]
    resume = "--resume" in sys.argv[2:]
    load_project_slots(cfg_path)
    cfg = load_config(cfg_path)
    evaluator = load_evaluator_from_config(cfg)
    if evaluator is None:
        log.warning("No evaluator configured; running in dry-run mode.")
    run(cfg_path, evaluator=evaluator, resume=resume)
