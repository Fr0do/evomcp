---
name: evolve
description: >
  Run GEPA (prompt/text evolution), EvoX (hyperparameter/patch evolution),
  or hybrid optimization using the evomcp MCP server. Use when the user wants
  to optimize prompts, rubrics, training hyperparameters, or code patches.

  MCP tools exposed: evolve_run, evolve_status, evolve_export,
  evolve_list_slots, evolve_inspect.

  Requires: evomcp MCP server configured in Claude Code settings.
---

# /evolve — GEPA + EvoX optimization

Two mutation surfaces, one evaluator. Pick the mode that matches what
the user wants to change:

| User intent | Mode | Mutates |
|-------------|------|---------|
| "Tune the critic rubric", "improve judge agreement", "fix prompt" | `gepa` | `text_genome` |
| "Sweep hyperparams", "try different clipping", "pick a patch" | `evox` | `prog_genome` |
| "Jointly optimize prompt + config" | `hybrid` | both |

## Commands

```
/evolve gepa configs/gepa.yaml          # GEPA-only on registered text slots
/evolve evox configs/evox.yaml          # EvoX-only on prog slots
/evolve hybrid configs/hybrid.yaml      # outer EvoX + inner GEPA
/evolve status                          # all recent runs + best-so-far
/evolve status <run_id>                 # single run
/evolve export <run_id>                 # export best candidate
/evolve slots [<project_root>]          # list registered search-space slots
/evolve inspect <bundle_dir>            # inspect a trace bundle
```

## How to execute (agent instructions)

1. Parse `$ARGUMENTS`:
   - `gepa|evox|hybrid <config>` → call `evolve_run(mode, config)`
   - `status [run_id]`           → call `evolve_status(run_id)`
   - `export <run_id>`           → call `evolve_export(run_id)`
   - `slots [dir]`               → call `evolve_list_slots(dir)`
   - `inspect <dir>`             → call `evolve_inspect(dir)`

2. Before starting a run, call `evolve_list_slots` and confirm the config's
   `target_text_slots` / `target_prog_slots` are all present.

3. Start the run with `background=true` (default). Report the `run_id`.

4. To watch progress, read the `log` path returned by `evolve_run` with
   the Monitor tool, filtering on:
   `round|generation|kappa=|best=|Early stop|ERROR|Traceback|GEPA complete|EvoX complete`

5. On completion, call `evolve_status(run_id)` then `evolve_export(run_id)`
   and report: primary_score, secondary_scores (kappa, latency), cost.

## Rules

- **Never mutate both surfaces in the same step.** GEPA touches only
  `text_genome`; EvoX touches only `prog_genome`.
- **Never pass free-form mutations.** The optimizer only writes registered
  slots (see `evolve_list_slots`).
- **On failure,** call `evolve_inspect` on the most recent trace bundle to
  diagnose before retrying. GEPA will only reflect on `bad_format` and
  `low_score` (semantically informative failures).
- **Commit after each run** that produces a better best-so-far. Include
  the `run_id` in the commit message so the archive maps to git history.

## MCP server setup (one-time)

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "evomcp": {
      "command": "evomcp",
      "args": ["serve"],
      "env": {
        "EVOMCP_ARTIFACTS": "/path/to/project/artifacts"
      }
    }
  }
}
```

Or if installed as a module:

```json
{
  "mcpServers": {
    "evomcp": {
      "command": "python",
      "args": ["-m", "evomcp.server"],
      "env": {
        "EVOMCP_ARTIFACTS": "/path/to/project/artifacts"
      }
    }
  }
}
```

## Project integration

Projects declare their own search spaces by creating:
```
optim/search_spaces/
  __init__.py     # imports prompts.py, configs.py, patches.py
  prompts.py      # registers TextSlot entries into DEFAULT_REGISTRY
  configs.py      # registers ProgSlot entries into DEFAULT_REGISTRY
  patches.py      # registers Patch + ProgSlot("patch_id") entries
```

And inherit the base protocol from `evomcp`:
```python
from evomcp.pipeline import Candidate, EvalResult, Budget
from evomcp.pipeline.registry import DEFAULT_REGISTRY
from evomcp.optim.evox_runner import run as evox_run
```

See [evomcp README](https://github.com/Fr0do/evomcp) for full docs.
