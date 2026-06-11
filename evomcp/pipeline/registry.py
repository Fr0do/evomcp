"""Text and program slot registries.

The optimizer never touches raw source files. It mutates named slots
declared here. This keeps the search space finite, inspectable, and
reversible.

TextSlot → one prompt / instruction / rule. GEPA mutates these.
ProgSlot → one typed knob (scalar, categorical, patch_id). EvoX mutates these.

Changes from initial scaffold (integrated from parameter-golf):
- TextSlot.mutatable: bool — quick on/off flag alongside fine-grained
  mutators_allowed. Lets a slot stay in the registry but be frozen for a
  specific run without removing it.
- resolve_patch_env() — looks up a Patch by id and returns its env_overrides
  dict. Used by pipeline.evaluator.materialize_prog_genome() so the evaluator
  never has to import evomcp.optim.search_spaces.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class SlotKind(str, Enum):
    TEXT = "text"
    SCALAR = "scalar"
    CATEGORICAL = "categorical"
    PATCH = "patch"


@dataclass(frozen=True)
class TextSlot:
    name: str
    role: str                                   # e.g. "planner_system_prompt"
    seed_value: str                             # baseline text
    description: str = ""
    max_chars: int = 8000
    mutatable: bool = True                      # False = frozen; GEPA skips this slot
    mutators_allowed: tuple[str, ...] = (       # which GEPA mutation types may touch this slot
        "rubric_text",
        "scale_definition",
        "scoring_examples",
        "cot_scaffolding",
    )


@dataclass(frozen=True)
class ProgSlot:
    name: str
    kind: SlotKind                              # SCALAR, CATEGORICAL, or PATCH
    default: Any
    bounds: tuple[float, float] | None = None   # SCALAR only
    choices: tuple[Any, ...] | None = None      # CATEGORICAL / PATCH only
    log_scale: bool = False                     # SCALAR only; sample in log space
    description: str = ""
    validator: Callable[[Any], bool] | None = None

    def sample_random(self, rng: random.Random | None = None) -> Any:
        """Sample a random value from this slot's domain.

        Used by EvoX to build initial population and to mutate genomes.
        """
        rng = rng or random.Random()
        if self.kind == SlotKind.SCALAR:
            lo, hi = self.bounds or (0.0, 1.0)
            if self.log_scale:
                import math
                return math.exp(rng.uniform(math.log(lo), math.log(hi)))
            return rng.uniform(lo, hi)
        elif self.kind in (SlotKind.CATEGORICAL, SlotKind.PATCH):
            return rng.choice(list(self.choices or [self.default]))
        return self.default


@dataclass
class SlotRegistry:
    text_slots: dict[str, TextSlot] = field(default_factory=dict)
    prog_slots: dict[str, ProgSlot] = field(default_factory=dict)
    # Patch env_overrides registry — populated by optim/search_spaces/patches.py
    _patch_env: dict[str, dict[str, Any]] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear all registered slots and patch envs.

        Project-specific search spaces are loaded dynamically into the shared
        registry, so repeated imports need an explicit reset to avoid duplicate
        registration failures and stale state leaking across runs.
        """
        self.text_slots.clear()
        self.prog_slots.clear()
        self._patch_env.clear()

    def register_text(self, slot: TextSlot) -> None:
        if slot.name in self.text_slots:
            raise ValueError(f"text slot already registered: {slot.name}")
        self.text_slots[slot.name] = slot

    def register_prog(self, slot: ProgSlot) -> None:
        if slot.name in self.prog_slots:
            raise ValueError(f"prog slot already registered: {slot.name}")
        self.prog_slots[slot.name] = slot

    def register_patch_env(self, patch_id: str, env: dict[str, Any]) -> None:
        """Register a patch's env_overrides. Called by search_spaces/patches.py."""
        self._patch_env[patch_id] = dict(env)

    def resolve_patch_env(self, patch_id: str) -> dict[str, Any]:
        """Return env_overrides for a patch_id without importing optim.search_spaces.

        This breaks the circular dependency: evaluator.materialize_prog_genome()
        calls DEFAULT_REGISTRY.resolve_patch_env() rather than importing the
        optim package (which itself imports pipeline).
        """
        if patch_id not in self._patch_env:
            raise KeyError(
                f"patch_id {patch_id!r} not in registry "
                f"(known: {sorted(self._patch_env)}). "
                "Did you import evomcp.optim.search_spaces before calling evaluate()?"
            )
        return dict(self._patch_env[patch_id])

    # --- genome sampling ---------------------------------------------------

    def sample_prog_genome(
        self,
        slots: list[str] | None = None,
        rng: random.Random | None = None,
    ) -> dict[str, Any]:
        """Random prog genome from the declared slots.

        From parameter-golf's sample_prog_genome(). Used by EvoX to build
        the initial population. If slots is None, samples all registered
        prog slots.
        """
        rng = rng or random.Random()
        target = slots or list(self.prog_slots)
        return {name: self.prog_slots[name].sample_random(rng) for name in target}

    def mutate_prog_genome(
        self,
        genome: dict[str, Any],
        slots: list[str] | None = None,
        rng: random.Random | None = None,
        n_mutations: int = 1,
    ) -> dict[str, Any]:
        """Single-step mutation: pick n_mutations random knobs, resample each.

        From parameter-golf's mutate_prog_genome(). EvoX calls this to
        generate offspring from an elite genome.
        """
        rng = rng or random.Random()
        target = slots or list(self.prog_slots)
        result = dict(genome)
        for name in rng.sample(target, min(n_mutations, len(target))):
            result[name] = self.prog_slots[name].sample_random(rng)
        return result

    # --- validation --------------------------------------------------------

    def validate_text_genome(self, genome: dict[str, str]) -> list[str]:
        """Return list of violation messages; empty = valid."""
        errs = []
        for name, value in genome.items():
            if name not in self.text_slots:
                errs.append(f"unknown text slot: {name}")
                continue
            slot = self.text_slots[name]
            if not isinstance(value, str):
                errs.append(f"{name}: expected str, got {type(value).__name__}")
            elif len(value) > slot.max_chars:
                errs.append(f"{name}: {len(value)} chars > max {slot.max_chars}")
        return errs

    def validate_prog_genome(self, genome: dict[str, Any]) -> list[str]:
        errs = []
        for name, value in genome.items():
            if name not in self.prog_slots:
                errs.append(f"unknown prog slot: {name}")
                continue
            slot = self.prog_slots[name]
            if slot.kind == SlotKind.SCALAR:
                if not isinstance(value, (int, float)):
                    errs.append(f"{name}: expected number, got {type(value).__name__}")
                elif slot.bounds is not None:
                    lo, hi = slot.bounds
                    if not (lo <= float(value) <= hi):
                        errs.append(f"{name}={value} outside [{lo}, {hi}]")
            elif slot.kind in (SlotKind.CATEGORICAL, SlotKind.PATCH):
                if slot.choices is None or value not in slot.choices:
                    errs.append(f"{name}={value!r} not in {slot.choices}")
            if slot.validator is not None and not slot.validator(value):
                errs.append(f"{name}: custom validator rejected {value!r}")
        return errs


# Module-level default registry. optim/search_spaces/*.py populate this.
DEFAULT_REGISTRY = SlotRegistry()
