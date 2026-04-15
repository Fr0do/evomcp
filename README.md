# evomcp

**GEPA + EvoX evolutionary optimizer — standalone Claude MCP server and skill.**

Evolves two mutation surfaces against a single evaluator:

| Surface | Optimizer | Mutates |
|---------|-----------|---------|
| `text_genome` | **GEPA** (reflective, trace-feedback, DSPy) | prompts, rubrics, reasoning scaffolds |
| `prog_genome` | **EvoX** (evolutionary, Pareto, CMA-ES/NSGA-II) | hyperparameters, architecture toggles, named patches |

Projects declare their own search spaces and evaluators; evomcp provides
the shared protocol, runners, and MCP interface.

## Install

```bash
pip install -e ~/experiments/evomcp
# or once published:
pip install evomcp
```

## Quick start

```bash
# Start the MCP server (add to Claude Code settings)
evomcp serve

# Or run directly from CLI
evomcp run gepa   my_project/configs/gepa.yaml
evomcp run evox   my_project/configs/evox.yaml
evomcp run hybrid my_project/configs/hybrid.yaml

evomcp status
evomcp export <run_id>
```

## Claude Code integration

1. Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "evomcp": {
      "command": "evomcp",
      "args": ["serve"],
      "env": { "EVOMCP_ARTIFACTS": "/path/to/project/artifacts" }
    }
  }
}
```

2. Copy or symlink `skills/SKILL.md` into your project's `skills/` directory
   so `/evolve` is available in Claude Code.

3. In Claude: `/evolve gepa configs/gepa.yaml`

## Project integration

Projects depend on evomcp and declare project-specific search spaces:

```python
# In your project's optim/search_spaces/prompts.py
from evomcp.pipeline.registry import DEFAULT_REGISTRY, TextSlot

DEFAULT_REGISTRY.register_text(TextSlot(
    name="my_critic_prompt",
    role="vlm_critic_system",
    seed_value="You are a critic...",
    description="System prompt for the VLM critic.",
))
```

```python
# In your project's pyproject.toml / requirements.txt
# evomcp @ file:///path/to/experiments/evomcp  (local)
# evomcp>=0.1.0  (once published)
```

See `hanfu-code` as a reference integration.

## MCP tools

| Tool | Description |
|------|-------------|
| `evolve_run(mode, config_path, resume, background)` | Start a run |
| `evolve_status(run_id)` | Check progress + best-so-far |
| `evolve_export(run_id)` | Export best candidate to artifacts/best/ |
| `evolve_list_slots(project_root)` | List registered search-space slots |
| `evolve_inspect(bundle_dir)` | Inspect a trace bundle |

## Protocol

Every evaluation writes a **trace bundle** under `artifacts/runs/<run_id>/traces/`:

```
<candidate_id[:12]>-seed<N>/
  candidate.json       Full Candidate (text + prog genome)
  budget.json          Budget applied (stage, limits, env_overrides)
  runtime_snapshot.json  git SHA, Python, pip hash, model versions
  inputs.json          Materialized env fed to subprocess
  events.jsonl         Line-delimited events (commands, tool calls, ...)
  stdout.log / stderr.log
  result.json          Full EvalResult
  replay.json          Minimal replay manifest
  failure.json         Only if error_type != None
```

GEPA reads `trace_feedback_summary(bundle_dir)` before each mutation step.
EvoX uses only `result.json` (primary + secondary scores).

## Architecture

```
evomcp/
  pipeline/        Canonical protocol
    candidate.py   Candidate, EvalResult, Budget, CostMetrics, FailureClass
    evaluator.py   Evaluator protocol, ScoringConfig, materialize_prog_genome()
    registry.py    SlotRegistry, TextSlot, ProgSlot (sample/mutate)
    metrics.py     Pareto front, weighted aggregation, constraints
    tracing.py     TraceBundle, RuntimeSnapshot, trace_feedback_summary()
  optim/
    gepa_runner.py   DSPy PromptMutation + Planner, TextParetoArchive
    evox_runner.py   EvoXState (resume), ParetoArchive, sample/mutate loop
    hybrid_runner.py Outer EvoX + gate-triggered inner GEPA
    search_spaces/   Empty base — projects add their own slots here
  server.py        FastMCP server (5 tools)
  cli.py           evomcp CLI
skills/
  SKILL.md         Claude Code /evolve skill
```

## Implementing an evaluator

```python
from evomcp.pipeline import Candidate, EvalResult, Budget
from evomcp.pipeline.evaluator import Evaluator
from pathlib import Path

class MyEvaluator:
    version = "v1"

    def evaluate(self, candidate: Candidate, budget: Budget, seed: int, *, run_dir: Path) -> EvalResult:
        # 1. materialize prog genome
        from evomcp.pipeline.evaluator import materialize_prog_genome
        from evomcp.pipeline.registry import DEFAULT_REGISTRY
        patch_env = DEFAULT_REGISTRY.resolve_patch_env(candidate.prog_genome.get("patch_id", "baseline"))
        env = materialize_prog_genome(candidate, budget, patch_env=patch_env)

        # 2. run your evaluation subprocess / API calls
        # 3. write trace bundle
        from evomcp.pipeline.tracing import TraceBundle, RuntimeSnapshot
        snap = RuntimeSnapshot.capture(budget.env_overrides.get("dataset_version", ""), self.version, seed)
        bundle = TraceBundle.create(run_dir, candidate, budget, snap)
        # ... evaluation ...
        result = EvalResult(candidate_id=candidate.candidate_id, success=True, primary_score=0.5)
        bundle.close(result)

        return result
```

## License

MIT
