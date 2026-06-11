"""Unit tests for evomcp.metrics.gore_captcha."""
from __future__ import annotations

import pytest

from evomcp.metrics.gore_captcha import (
    call_chain_f1,
    char_f1,
    exact_match,
    exec_validity,
    score_gore_captcha,
    step_prm_mean,
)


# ---------------------------------------------------------------------------
# Sub-metric primitives
# ---------------------------------------------------------------------------

def test_exact_match_normalises_whitespace_and_case():
    assert exact_match("  ALPHA77 ", "alpha77")
    assert not exact_match("ALPHA78", "ALPHA77")
    assert not exact_match(None, "ALPHA77")
    assert not exact_match("", "ALPHA77")


def test_char_f1_happy_path():
    # Anagram: perfect recall + precision.
    assert char_f1("77AHPLA", "ALPHA77") == pytest.approx(1.0)


def test_char_f1_partial():
    # Replace one char: 6/7 match.
    score = char_f1("ALPHA78", "ALPHA77")
    assert 0.7 < score < 1.0


def test_char_f1_empty_both_is_perfect():
    assert char_f1("", "") == 1.0


def test_char_f1_one_side_empty_is_zero():
    assert char_f1("", "ALPHA77") == 0.0
    assert char_f1("ALPHA77", "") == 0.0


# ---------------------------------------------------------------------------
# Call chain F1
# ---------------------------------------------------------------------------

def _mk_extcall(fn, payload_suffix=""):
    return {"type": "EXTCALL", "env": {}, "payload": f"X = @{fn}({payload_suffix}) → ..."}


def test_call_chain_f1_perfect():
    gold = [_mk_extcall("reverse"), _mk_extcall("upper")]
    pred = [_mk_extcall("reverse"), _mk_extcall("upper")]
    assert call_chain_f1(pred, gold) == 1.0


def test_call_chain_f1_wrong_order():
    gold = [_mk_extcall("reverse"), _mk_extcall("upper")]
    pred = [_mk_extcall("upper"), _mk_extcall("reverse")]
    # Positional F1 → 0/2 match.
    assert call_chain_f1(pred, gold) == 0.0


def test_call_chain_f1_goreeval_string_traces():
    gold = ["ECALL X = @reverse(\"ALPHA77\") → \"77AHPLA\"",
            "ECALL Y = @upper(X) → \"77AHPLA\""]
    pred = ["ECALL X = @reverse(\"ALPHA77\") → \"77AHPLA\"",
            "ECALL Y = @upper(X) → \"77AHPLA\""]
    assert call_chain_f1(pred, gold) == 1.0


def test_call_chain_f1_mixed_traces_type():
    """Dict trace and string trace should parse to same @fn sequence."""
    dict_trace = [_mk_extcall("reverse"), _mk_extcall("upper")]
    str_trace = ["ECALL X = @reverse(\"ALPHA77\") → \"77AHPLA\"",
                 "ECALL Y = @upper(X) → \"77AHPLA\""]
    assert call_chain_f1(dict_trace, str_trace) == 1.0


# ---------------------------------------------------------------------------
# Exec validity
# ---------------------------------------------------------------------------

def test_exec_validity_full_credit():
    pred = {"trace": [_mk_extcall("reverse")], "error": None}
    assert exec_validity(pred) == 1.0


def test_exec_validity_hard_error_zero():
    pred = {"trace": [_mk_extcall("reverse")], "error": "exec_error"}
    assert exec_validity(pred) == 0.0


def test_exec_validity_no_trace_zero():
    assert exec_validity({"trace": [], "error": None}) == 0.0


def test_exec_validity_non_gore_trace_halfcredit():
    # Trace exists but tag isn't a GORE primitive.
    assert exec_validity({"trace": ["FOO something"], "error": None}) == 0.5


# ---------------------------------------------------------------------------
# Step PRM mean (mirrors gorevm/src/score.rs)
# ---------------------------------------------------------------------------

def test_step_prm_mean_exact_both_empty():
    assert step_prm_mean([], []) == 1.0


def test_step_prm_mean_exact_match():
    trace = [_mk_extcall("reverse")]
    assert step_prm_mean(trace, trace) == 1.0


def test_step_prm_mean_type_mismatch():
    gold = [{"type": "EXTCALL", "env": {}, "payload": "p"}]
    pred = [{"type": "STEP", "env": {}, "payload": "p"}]
    assert step_prm_mean(pred, gold) == -0.5


def test_step_prm_mean_env_mismatch():
    gold = [{"type": "EXTCALL", "env": {"X": "a"}, "payload": "p"}]
    pred = [{"type": "EXTCALL", "env": {"X": "b"}, "payload": "p"}]
    assert step_prm_mean(pred, gold) == pytest.approx(0.3)


def test_step_prm_mean_payload_mismatch():
    gold = [{"type": "EXTCALL", "env": {"X": "a"}, "payload": "p1"}]
    pred = [{"type": "EXTCALL", "env": {"X": "a"}, "payload": "p2"}]
    assert step_prm_mean(pred, gold) == pytest.approx(0.7)


def test_step_prm_mean_length_penalty():
    gold = [_mk_extcall("reverse"), _mk_extcall("upper")]
    pred = [_mk_extcall("reverse")]
    # aligned=1 exact (1.0), 1 missing (-0.5) → mean = 0.25
    assert step_prm_mean(pred, gold) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Top-level composite scorer
# ---------------------------------------------------------------------------

def _gold_alpha77():
    return {
        "key": "ALPHA77",
        "trace": [
            {"type": "EXTCALL", "env": {"X": "\"77AHPLA\""},
             "payload": "X = @reverse(\"77AHPLA\") → \"ALPHA77\""},
            {"type": "EXTCALL", "env": {"X": "\"77AHPLA\"", "Y": "\"ALPHA77\""},
             "payload": "Y = @upper(X) → \"ALPHA77\""},
        ],
    }


def test_score_perfect_prediction():
    gold = _gold_alpha77()
    pred = {"key": "alpha77", "trace": gold["trace"], "error": None}
    s = score_gore_captcha(gold, pred)
    assert s.primary_score == pytest.approx(1.0)
    assert s.secondary_scores["key_em"] == 1.0
    assert s.failure_class is None


def test_score_empty_prediction():
    s = score_gore_captcha(_gold_alpha77(), {})
    assert s.primary_score == 0.0
    assert s.failure_class == "empty"


def test_score_key_only_no_trace():
    """Model recovered the key somehow, but produced no trace.
    Should still get credit for key accuracy, zero for trace components.
    """
    gold = _gold_alpha77()
    pred = {"key": "ALPHA77", "trace": [], "error": None}
    s = score_gore_captcha(gold, pred)
    # 0.4*1.0 (char_f1) + 0.1*1.0 (em) = 0.5
    # All other components zero, step_prm after re-map: (1.0 for empty-vs-nonempty? no,
    # different lens → penalty. Here gold has 2 nodes, pred has 0 → mean -0.5, re-mapped 0.0)
    assert s.secondary_scores["key_em"] == 1.0
    assert s.secondary_scores["key_char_f1"] == 1.0
    assert s.primary_score == pytest.approx(0.5)


def test_score_near_miss_key():
    """One-character-off prediction with correct call chain."""
    gold = _gold_alpha77()
    pred = {"key": "ALPHA78", "trace": gold["trace"], "error": None}
    s = score_gore_captcha(gold, pred)
    assert 0.6 < s.primary_score < 1.0
    assert s.secondary_scores["key_em"] == 0.0
    assert s.secondary_scores["call_chain_f1"] == 1.0


def test_score_wrong_call_chain_but_right_key():
    """Model stumbled onto the key with a divergent call chain."""
    gold = _gold_alpha77()
    bad_trace = [{"type": "EXTCALL", "env": {}, "payload": "X = @lower(...) → ..."}]
    pred = {"key": "ALPHA77", "trace": bad_trace, "error": None}
    s = score_gore_captcha(gold, pred)
    # call_chain_f1 = 0 (upper vs lower, mis-aligned length)
    assert s.secondary_scores["call_chain_f1"] == 0.0
    assert s.secondary_scores["key_em"] == 1.0
    # 0.15·exec + 0.40·char_f1 + 0.20·prm_remap + 0.10·em ≈ 0.70
    assert 0.6 < s.primary_score < 0.75


def test_score_secondary_keys_present():
    s = score_gore_captcha(_gold_alpha77(),
                           {"key": "X", "trace": [], "error": None})
    for k in ("call_chain_f1", "exec_validity", "key_char_f1",
              "step_prm_mean_raw", "key_em"):
        assert k in s.secondary_scores, k
