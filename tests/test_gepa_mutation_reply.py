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
