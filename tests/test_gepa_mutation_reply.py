"""Robustness tests for parsing the GEPA mutation LM reply.

Small models (e.g. openrouter haiku) often wrap JSON in ```json fences,
prepend prose, or return an empty string. A bare json.loads then dies with
"Expecting value: line 1 column 1 (char 0)" and kills the whole mutation
step. _loads_mutation_reply must recover the JSON object regardless.
"""
import pytest

from evomcp.optim.gepa_runner import _loads_mutation_reply


def test_plain_json_object():
    assert _loads_mutation_reply('{"slot": "text"}') == {"slot": "text"}


def test_json_fenced():
    assert _loads_mutation_reply('```json\n{"slot": "text"}\n```') == {"slot": "text"}


def test_bare_fence_without_lang():
    assert _loads_mutation_reply('```\n{"slot": "text"}\n```') == {"slot": "text"}


def test_leading_prose_then_fence():
    reply = 'Here is the revised JSON:\n```json\n{"a": "x", "b": "y"}\n```\nDone.'
    assert _loads_mutation_reply(reply) == {"a": "x", "b": "y"}


def test_inline_prose_around_object():
    assert _loads_mutation_reply('Sure! {"a": "x"} hope that helps') == {"a": "x"}


@pytest.mark.parametrize("bad", ["", "   ", None, "no json here", "[1, 2, 3]"])
def test_unrecoverable_raises_valueerror(bad):
    with pytest.raises(ValueError):
        _loads_mutation_reply(bad)


def test_slot_prefixed_fallback():
    # The format an older signature docstring requested: `slot: <text>`.
    reply = (
        "paper2spec_system: You are a reproduction engineer.\n"
        "Work only from the evidence.\n"
        "paper2spec_claims: Decompose into checkable claims."
    )
    out = _loads_mutation_reply(
        reply, slot_names=["paper2spec_system", "paper2spec_claims"])
    assert set(out) == {"paper2spec_system", "paper2spec_claims"}
    assert out["paper2spec_system"].startswith("You are a reproduction engineer")
    assert "Work only from the evidence." in out["paper2spec_system"]
    assert out["paper2spec_claims"] == "Decompose into checkable claims."


def test_slot_prefixed_needs_known_slots():
    # Without slot_names, a non-JSON slot blob is unrecoverable.
    with pytest.raises(ValueError):
        _loads_mutation_reply("paper2spec_system: text here")


def test_json_preferred_over_slot_fallback():
    # A valid JSON object is returned even when slot_names are provided.
    out = _loads_mutation_reply('{"s": "v"}', slot_names=["s"])
    assert out == {"s": "v"}
