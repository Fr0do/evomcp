---
name: evolve
description: >
  Run GEPA (prompt/text evolution), EvoX (hyperparameter/patch evolution),
  or hybrid optimization using the evomcp MCP server. Use when the user wants
  to optimize prompts, rubrics, training hyperparameters, or code patches.

  MCP tools exposed: evolve_run, evolve_status, evolve_export,
  evolve_list_slots, evolve_inspect.

  Requires: evomcp MCP server configured in Codex or Claude Code MCP settings.
---

# evolve — GEPA + EvoX optimization

Two mutation surfaces, one evaluator. Pick the mode that matches what
the user wants to change:

| User intent | Mode | Mutates |
|-------------|------|---------|
| "Tune the critic rubric", "improve judge agreement", "fix prompt" | `gepa` | `text_genome` |
| "Sweep hyperparams", "try different clipping", "pick a patch" | `evox` | `prog_genome` |
| "Jointly optimize prompt + config" | `hybrid` | both |

## Example asks

```
Run GEPA with configs/gepa.yaml
Run EvoX with configs/evox.yaml
Run hybrid optimization with configs/hybrid.yaml
Show evolve status
Export run <run_id>
List evolve slots for <project_root>
Inspect evolve bundle <bundle_dir>
```

## MCP tool signatures (verified against server.py)

```
evolve_run(mode: str, config_path: str,
           resume: bool = False, background: bool = True) -> JSON
  # Background → {run_id, status: "running", log, hint}
  # Foreground → {run_id, status: "complete"|"error", stdout_tail, stderr_tail}
  # `config_path` is a filesystem path to a YAML file (not a YAML blob).

evolve_status(run_id: str = "") -> JSON
  # Empty run_id → {runs: [...], total}
  # Specific run → enriched entry with best_pareto_archive,
  #                best_text_pareto_archive (each: primary_score, candidate_id)

evolve_export(run_id: str) -> JSON
  # {run_id, exported_to, best_prog?, best_text?}
  # Each best_*: {primary_score, candidate_id, path}

evolve_list_slots(project_root: str = ".") -> JSON
  # {project_root, text_slots: {...}, prog_slots: {...}}

evolve_inspect(bundle_dir: str) -> JSON
  # {summary, candidate_id, stage, evaluator_version, git_sha}
  # `bundle_dir` is a trace bundle path, NOT a run_id.
```

## How to execute (agent instructions)

1. Parse the user request:
   - `gepa|evox|hybrid <config>` → call `evolve_run(mode, config_path)`
   - `status [run_id]`           → call `evolve_status(run_id or "")`
   - `export <run_id>`           → call `evolve_export(run_id)`
   - `slots [dir]`               → call `evolve_list_slots(dir or ".")`
   - `inspect <dir>`             → call `evolve_inspect(bundle_dir)`

2. Before starting a run, call `evolve_list_slots(project_root)` and confirm the
   config's `target_text_slots` / `target_prog_slots` are all present.

3. Read `budget.max_cost_usd` from the YAML and abort with an ask-the-user if
   absent (see "Cost ceiling" below).

4. Start the run with `background=True` (default). Report the `run_id`.

5. To watch progress, read the `log` path returned by `evolve_run` from the
   workspace and look for:
   `round|generation|kappa=|best=|Early stop|ERROR|Traceback|GEPA complete|EvoX complete`

6. On completion, call `evolve_status(run_id)` then `evolve_export(run_id)`
   and report: `best_pareto_archive.primary_score` and `best_text_pareto_archive.primary_score`
   (both may be absent — check each key). Detailed cost / secondary metrics
   are in the per-archive JSON files under `artifact_dir`, not surfaced by
   `evolve_status` itself.

## evolve_inspect usage

Call when a run fails or produces a confusing result. Expected use:

```
inspect <artifact_dir>/traces/<bundle_name>
```

Returns `{summary, candidate_id, stage, evaluator_version, git_sha}`.
`summary` is the compact `trace_feedback_summary` dict — counts of
successes, errors, events. No GEPA-specific "reflections" field;
reflections are internal to the runner.

Cheap — does not re-run anything. Call before any retry.

## Choosing mutation_backend

| Situation                              | backend    |
|----------------------------------------|------------|
| Default, cloud-available               | `claude`   |
| Remote GPU via SSH                     | `ssh_vllm` |
| Local GPU, OpenAI-compatible endpoint  | `vllm`     |
| Cross-model comparison                 | `openrouter` |
| OpenAI-specific features               | `openai`   |

YAML block shape (under `gepa.mutation_backend` or `inner_gepa.mutation_backend`):

```yaml
mutation_backend:
  backend: ssh_vllm
  model: Qwen/Qwen3-32B-Instruct
  ssh_host: kurkin-4          # any alias from ~/.ssh/config
  remote_port: 8000           # default
  max_tokens: 4096            # default
```

For `ssh_vllm`, the runner auto-opens an SSH tunnel, picks a free local
port, and tears it down on process exit. Preflight:
`ssh -o ConnectTimeout=5 -o BatchMode=yes <host> true` — fails fast if
unreachable. The agent should verify reachability itself before a long run.

For `openrouter`, set `OPENROUTER_API_KEY` in the env or pass `api_key`
in the YAML block.

## Cost ceiling

All configs must declare `budget.max_cost_usd`. The agent must:

1. Read it before calling `evolve_run`.
2. Refuse and ask the user for a ceiling if absent.
3. Warn whenever it polls `evolve_status` and observes cost approaching
   the ceiling (cost data lives in the per-archive JSON under
   `artifact_dir`, not in the `evolve_status` response — read the file).

Defaults by mode (override per-config):

| Mode   | Default ceiling | Rationale |
|--------|-----------------|-----------|
| gepa   | $5              | ~500 rollouts × ~$0.01/Haiku call |
| evox   | $2              | No LM calls per candidate — mostly compute |
| hybrid | $10             | Outer EvoX + inner GEPA |

**Caveat:** `background=True` launches the run as a subprocess; the MCP
tool layer cannot halt it mid-run. Enforcement lives inside the runner.
The agent can only observe and warn — not auto-halt — unless it kills
the pid from `evolve_status`.

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

### Codex (codex-cli)

Add to your Codex MCP configuration (`~/.codex/config.yaml` or project
`codex.yaml`):

```yaml
mcp_servers:
  evomcp:
    command: evomcp
    args: [serve]
    env:
      EVOMCP_ARTIFACTS: /path/to/project/artifacts
```

Or, using the JSON form:

```json
{
  "mcp_servers": {
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

### Claude Code

Add to `~/.claude/config.json` (or the project-level `.mcp.json`):

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

### Module form (either host)

If `evomcp` isn't on PATH, substitute:

```json
"command": "python",
"args": ["-m", "evomcp.server"]
```

The `EVOMCP_ARTIFACTS` env var is the single source of truth for run
discovery across server restarts — point it at a stable per-project dir.

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

The server auto-loads project `optim/search_spaces` from the config path,
so `evolve_list_slots` and `evolve_run` work without manually importing
slots first.
