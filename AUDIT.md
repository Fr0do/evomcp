# evomcp Phase 1 audit

Generated for the GEPA+EvoX enhancement plan. Compares `skills/SKILL.md` claims
against actual `evomcp/server.py` and runner code at HEAD (branch: `main`,
working tree dirty — uncommitted changes staged for this enhancement pass).

## 1. Actual MCP tool signatures

From `evomcp/server.py` (FastMCP registrations at lines 337, 358, 370, 382, 394):

| Tool | Signature | Returns (JSON) |
|------|-----------|----------------|
| `evolve_run` | `(mode: str, config_path: str, resume: bool = False, background: bool = True) -> str` | Background: `{run_id, status: "running", log, hint}`. Foreground: `{run_id, status: "complete"\|"error", stdout_tail, stderr_tail}` |
| `evolve_status` | `(run_id: str = "") -> str` | Empty `run_id` → `{runs: [...], total}`. Specific → enriched entry with optional `best_pareto_archive`, `best_text_pareto_archive` |
| `evolve_export` | `(run_id: str) -> str` | `{run_id, exported_to, best_prog?, best_text?}` — each `best_*` is `{primary_score, candidate_id, path}` |
| `evolve_list_slots` | `(project_root: str = ".") -> str` | `{project_root, text_slots: {name: {role, description, mutatable, max_chars}}, prog_slots: {name: {kind, default, bounds, choices, log_scale, description}}}` |
| `evolve_inspect` | `(bundle_dir: str) -> str` | `{summary, candidate_id, stage, evaluator_version, git_sha}` |

All tools return JSON **strings**, not dicts (FastMCP wraps via `json.dumps`).

## 2. SKILL.md drift

### 2a. Correct claims (verified)

- `evolve_run` accepts `background` with default `True` — SKILL.md's "Start the run with `background=true` (default)" is **accurate**. (server.py:338-342, 342)
- `evolve_inspect` takes a path (`bundle_dir`), not a `run_id` — SKILL.md is **accurate**. (server.py:395)
- Tool name set `{evolve_run, evolve_status, evolve_export, evolve_list_slots, evolve_inspect}` matches exactly. (server.py:337-404)

### 2b. Incorrect / incomplete claims

| # | SKILL.md claim | Reality | Severity |
|---|-----------------|---------|----------|
| D1 | "`evolve_run(mode, config)`" (execution-rules section, line 41) | Actual second arg is **`config_path`** — a filesystem path, **not** a dict or YAML blob. The server validates `cfg.exists()` (server.py:83-85). | HIGH — misleads the agent about what to pass |
| D2 | "On completion... report: primary_score, secondary_scores (kappa, latency), cost" (line 56) | `evolve_status` only returns `best_*.primary_score` and `candidate_id`. **No `secondary_scores`, no `cost_usd`** in the enriched entry. (server.py:170-196) | HIGH — the skill promises data the server does not return |
| D3 | `evolve_status` "run_id" is the param (line 42) | Default is `""` (empty string), not absent. Server maps `""` → None internally (server.py:368). Minor but worth stating. | LOW |
| D4 | `evolve_inspect` returns "reflections" / GEPA verbal gradients (implied by the "diagnose before retrying" framing, line 62) | Actual return is `{summary, candidate_id, stage, evaluator_version, git_sha}`. The `summary` is a `trace_feedback_summary` dict — not an explicit `reflections` field. (server.py:302-309) | MED — wording suggests a richer payload than exists |
| D5 | No cost-ceiling mechanism mentioned or enforced anywhere | Confirmed: no `budget.max_cost_usd` check in `server.py` or runners. Cost is tracked in `EvalResult.cost` but never gated. | HIGH — Phase 3 needs to add this to both SKILL.md and optionally to server |
| D6 | Codex-only MCP setup block | No mirror for `~/.claude/config.json` (Claude Code host). Same server works in both. | MED |

### 2c. Non-drift, but missing from SKILL.md

- `evolve_run` also accepts `resume: bool = False` — not mentioned. (server.py:341)
- Server uses `EVOMCP_ARTIFACTS` env var for run registry + exports. SKILL.md's setup block shows it, but doesn't state it's the single source of truth for run discovery across restarts.
- `_enrich_entry` liveness-checks the subprocess PID (server.py:177-181) — restart-safe status. Worth noting so the agent doesn't assume stale status after a server restart.

## 3. Hardcoded mutation-backend assumptions

Only two call sites, both with identical branch logic:

| File | Line | Code |
|------|------|------|
| `evomcp/optim/gepa_runner.py` | 219 | `lm_model = mutation_backend.get("model", "claude-haiku-4-5")` |
| `evomcp/optim/gepa_runner.py` | 220 | `lm_backend = mutation_backend.get("backend", "claude")` |
| `evomcp/optim/gepa_runner.py` | 224-229 | `if lm_backend == "claude": ... elif "openai": ... else: dspy.LM(lm_model, ...)` |
| `evomcp/optim/hybrid_runner.py` | 365-366 | same defaults |
| `evomcp/optim/hybrid_runner.py` | 368-373 | same branch pattern (duplicate of gepa_runner) |

**No other hits.** `evox_runner.py` does no LM calls — it's genome sampling + evaluator dispatch only. This means Phase 2 only needs to refactor two sites, and inner-GEPA in `hybrid_runner._run_inner_gepa` shares the same logic as the outer GEPA entry point.

Default model `claude-haiku-4-5` and default backend `claude` are duplicated. Phase 2 dispatcher should centralize both defaults.

## 4. Hardcoded hostnames / endpoints

**None.** Full sweep for `kurkin|localhost:\d|127\.0\.0\.1` returned only `.omc/project-memory.json` (a path, not a host). The codebase is clean for Phase 2 SSH work — no existing wiring to undo.

## 5. Other observations relevant to Phase 2/3

- `evolve_status._enrich_entry` is the natural place to surface cost. `EvalResult.cost` (CostMetrics) is already written to per-run archive files. Adding a cost pass is ~10 lines.
- `hybrid_runner._run_inner_gepa` (line 340-439) duplicates the dspy config block from `gepa_runner.run`. A shared `build_mutation_lm(config)` dispatcher eliminates this duplication as a side effect of Phase 2.
- The `else:` branch at `gepa_runner.py:229` and `hybrid_runner.py:373` is currently the only path for anything that isn't `claude` or `openai`. It passes the raw model string to `dspy.LM`, which will fail for any model that needs a base_url (including vLLM). This is why Phase 2's `vllm` backend is a genuine new capability, not just a rename.
- `background=True` subprocess model works, but the parent `server.py` has no way to signal cost-ceiling violation to a running child. Enforcement has to happen inside the runner process, not in the MCP tool layer. Phase 3 docs should reflect this — the agent must poll `evolve_status` for the warning; the run won't auto-halt unless the runner is modified.

## 6. Phase 2 scope confirmation

Given the audit, Phase 2 is narrower than the plan implies:

- Refactor sites: **2** (gepa_runner.py:219-229, hybrid_runner.py:365-373). Plus `_run_inner_gepa` dedupe, also in hybrid_runner.
- New module: `evomcp/backends/mutation.py` with 5 classes + `build_mutation_lm`.
- Preflight check for `ssh_vllm`: `ssh -o ConnectTimeout=5 -o BatchMode=yes <host> true` — one `subprocess.run`.
- Tunnel lifecycle: `atexit`-registered teardown, cached per `(host, remote_port)` key.
- No changes needed in `server.py` for Phase 2 itself; backend choice is per-config YAML and flows through to the runner unchanged.

## 7. Phase 3 scope confirmation

Given D1, D2, D3, D4, D5, D6 above, SKILL.md needs edits at these locations:

- Line 41 — fix `config` → `config_path` with "path to YAML file" note.
- Line 56 — replace the "report primary_score, secondary_scores, cost" claim with the actual `best_*.primary_score` / `candidate_id` shape, OR extend `_enrich_entry` in server.py to surface cost (recommended; ~10 LOC).
- After line 62 — add an explicit "### evolve_inspect usage" block describing the actual return keys.
- Add a "### Choosing mutation_backend" block (claude / ssh_vllm / vllm / openrouter) — nothing like it exists.
- Add a "### Cost ceiling" block — nothing like it exists. Note the enforcement caveat from §5.
- Mirror the Codex MCP setup JSON for `~/.claude/config.json` — Claude Code parity.

## 8. Recommendation

Phase 2 and Phase 3 are both smaller than the plan estimated. The `ssh_vllm` tunnel class is the only piece with real implementation risk (process lifecycle, port allocation, teardown on abnormal exit). Everything else is mechanical.

Proceed to Phase 2. No blockers found.
