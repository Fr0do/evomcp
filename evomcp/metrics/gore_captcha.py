"""GORE-adapted CIPHER captcha metric.

Problem shape:
    A "CIPHER captcha" is a GORE program whose body is a chain of ExtCall
    primitives (`@reverse`, `@upper`, `@lower`, `@concat`, `@nth`, ...) applied
    to a known obfuscated key. The solver must execute the chain and recover
    the plain-text gold key.

Gold payload (one JSONL line):
    {
        "key":    "ALPHA77",                 # plain-text ground truth
        "ops":    ["reverse", "upper"],      # CALL chain in application order
        "program": "<gore source>",          # .gore source
        "query":   "solve(X)",
        "trace":   [TraceNode, ...],         # gold GORE interpreter trace
    }

Predicted payload:
    {
        "key":    "77AHPLA" | "" | None,     # model's recovered key
        "trace":  [TraceNode, ...],          # model's emitted trace (can be empty)
        "solutions": [{"X": "\"ALPHA77\""}], # optional, parsed from model output
        "error":  str | None,                # runtime error class if any
    }

Score composition (aligned with REVEL pipeline_F1 in captcha-bench but
re-weighted for GORE's step-level signal):

    primary = 0.15 * S1_call_chain_f1
            + 0.15 * S2_exec_validity
            + 0.40 * S3_key_char_f1
            + 0.20 * S4_step_prm_mean
            + 0.10 * S5_key_em

All sub-scores land in [0, 1] after clipping; `step_prm_mean` is re-mapped
from [-0.5, 1.0] to [0, 1] before weighting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# PRM reward values mirror gorevm/src/score.rs (Definition 1 of the GORE paper).
PRM_TYPE_MISMATCH = -0.5
PRM_LEN_MISMATCH = -0.5
PRM_ENV_MISMATCH = 0.3
PRM_PAYLOAD_MISMATCH = 0.7
PRM_EXACT = 1.0

# Node types recognised in GORE traces (UPPERCASE, matching gorevm serde tags).
_NODE_TYPES = {"STEP", "FORK", "CUT", "LET", "CALL", "EXTCALL"}


@dataclass(frozen=True)
class CaptchaScore:
    primary_score: float
    secondary_scores: dict[str, float]
    failure_class: str | None  # one of: None | "empty" | "no_key" | "exec_error"


# ---------------------------------------------------------------------------
# Sub-metrics
# ---------------------------------------------------------------------------

def _norm_key(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip().upper()


def exact_match(pred_key: Any, gold_key: Any) -> bool:
    p, g = _norm_key(pred_key), _norm_key(gold_key)
    return bool(p) and p == g


def char_f1(pred_key: Any, gold_key: Any) -> float:
    """Multiset character F1 — partial credit for near-misses on captcha keys."""
    p, g = list(_norm_key(pred_key)), list(_norm_key(gold_key))
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = sum(min(p.count(c), g.count(c)) for c in set(g))
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)


def _extcall_sequence(trace: list[dict]) -> list[str]:
    """Extract the ordered list of `@fn` names actually invoked in a trace.

    TraceNode shape (from gorevm): {"type": "EXTCALL", "payload": "X = @fn(...) → ..."}.
    Tolerates JSON-typed-as-dict AND string payloads (goreeval's format).
    """
    calls: list[str] = []
    for node in trace or []:
        if isinstance(node, str):
            # goreeval-style: single-line human strings. We match either
            # "CALL X = @fn(..." or "ECALL X = @fn(..."
            s = node.lstrip()
            if s.startswith(("CALL ", "ECALL ")):
                at = s.find("@")
                if at != -1:
                    rest = s[at + 1:]
                    paren = rest.find("(")
                    if paren != -1:
                        calls.append(rest[:paren].strip())
            continue
        ntype = str(node.get("type", "")).upper()
        if ntype not in ("EXTCALL", "CALL"):
            continue
        payload = str(node.get("payload", ""))
        at = payload.find("@")
        if at == -1:
            continue
        rest = payload[at + 1:]
        paren = rest.find("(")
        if paren == -1:
            continue
        calls.append(rest[:paren].strip())
    return calls


def call_chain_f1(pred_trace: list[dict], gold_trace: list[dict]) -> float:
    """Sequence-aware F1 on the ExtCall chain (REVEL S1 adapted)."""
    p = _extcall_sequence(pred_trace)
    g = _extcall_sequence(gold_trace)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    correct = sum(pi == gi for pi, gi in zip(p, g))
    prec = correct / len(p) if p else 0.0
    rec = correct / len(g) if g else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def exec_validity(pred: dict) -> float:
    """S2 analogue: 1.0 iff the solver ran to completion with ≥1 trace node
    and no structural error. Partial 0.5 credit if a trace exists but error
    set to a soft class (e.g. `low_score` — still exercised primitives)."""
    if pred.get("error") in ("exec_error", "unknown_external", "type_error", "panic"):
        return 0.0
    trace = pred.get("trace") or []
    if not trace:
        return 0.0
    # Must contain at least one structured primitive.
    if isinstance(trace[0], dict):
        types_seen = {str(n.get("type", "")).upper() for n in trace}
    else:
        types_seen = {s.split(" ", 1)[0] for s in trace if isinstance(s, str)}
    return 1.0 if types_seen & _NODE_TYPES else 0.5


def _node_type_of(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("type", "")).upper()
    if isinstance(node, str):
        head = node.strip().split(" ", 1)[0]
        # Normalise goreeval's "ECALL" → "EXTCALL" for cross-evaluator parity.
        return "EXTCALL" if head == "ECALL" else head.upper()
    return ""


def _node_env(node: Any) -> Any:
    if isinstance(node, dict):
        return node.get("env")
    return None  # string traces don't carry env → treat as equal-unless-typed-differently


def _node_payload(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("payload", ""))
    if isinstance(node, str):
        # Drop the leading tag, keep the rest as payload.
        parts = node.strip().split(" ", 1)
        return parts[1] if len(parts) > 1 else ""
    return ""


def step_prm_mean(pred_trace: list[Any], gold_trace: list[Any]) -> float:
    """Port of gorevm/src/score.rs::score — aligned-node PRM mean.

    Returns the raw mean reward in [-0.5, 1.0]. Caller re-maps to [0, 1]
    before weighting into the composite score.
    """
    g_len = len(gold_trace or [])
    p_len = len(pred_trace or [])
    n = max(g_len, p_len)
    if n == 0:
        return PRM_EXACT  # both empty traces = trivially exact

    aligned = min(g_len, p_len)
    total = 0.0
    for i in range(aligned):
        g, p = gold_trace[i], pred_trace[i]
        gt, pt = _node_type_of(g), _node_type_of(p)
        if gt != pt:
            total += PRM_TYPE_MISMATCH
        elif _node_env(g) != _node_env(p):
            total += PRM_ENV_MISMATCH
        elif _node_payload(g) != _node_payload(p):
            total += PRM_PAYLOAD_MISMATCH
        else:
            total += PRM_EXACT
    total += PRM_LEN_MISMATCH * (n - aligned)
    return total / n


# ---------------------------------------------------------------------------
# Top-level scorer
# ---------------------------------------------------------------------------

def score_gore_captcha(gold: dict, pred: dict) -> CaptchaScore:
    """Compose a GORE-native CIPHER captcha score from (gold, pred).

    See module docstring for payload shape. Returns a `CaptchaScore` with
    `primary_score` in [0, 1], a `secondary_scores` dict of sub-metrics, and
    `failure_class` tagged for GEPA's reflection filter.
    """
    pred = pred or {}
    gold = gold or {}

    if not pred or (pred.get("key") is None and not pred.get("trace")):
        return CaptchaScore(
            primary_score=0.0,
            secondary_scores={
                "call_chain_f1": 0.0,
                "exec_validity": 0.0,
                "key_char_f1": 0.0,
                "step_prm_mean_raw": PRM_TYPE_MISMATCH,
                "key_em": 0.0,
            },
            failure_class="empty",
        )

    s1 = call_chain_f1(pred.get("trace", []), gold.get("trace", []))
    s2 = exec_validity(pred)
    s3 = char_f1(pred.get("key"), gold.get("key"))
    s4_raw = step_prm_mean(pred.get("trace", []), gold.get("trace", []))
    s4 = (s4_raw - PRM_TYPE_MISMATCH) / (PRM_EXACT - PRM_TYPE_MISMATCH)
    s4 = max(0.0, min(1.0, s4))
    s5 = 1.0 if exact_match(pred.get("key"), gold.get("key")) else 0.0

    primary = (
        0.15 * s1
        + 0.15 * s2
        + 0.40 * s3
        + 0.20 * s4
        + 0.10 * s5
    )
    primary = max(0.0, min(1.0, primary))

    failure: str | None = None
    if primary == 0.0:
        failure = "no_key" if not pred.get("key") else "exec_error"

    return CaptchaScore(
        primary_score=primary,
        secondary_scores={
            "call_chain_f1": s1,
            "exec_validity": s2,
            "key_char_f1": s3,
            "step_prm_mean_raw": s4_raw,
            "key_em": s5,
        },
        failure_class=failure,
    )
