"""Unit tests for evomcp.backends.mutation."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_dspy(monkeypatch):
    """Install a fake dspy module that records LM() calls."""
    mod = MagicMock()

    class FakeLM:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    mod.LM = FakeLM
    monkeypatch.setitem(sys.modules, "dspy", mod)
    yield mod


def test_dispatcher_unknown_backend_raises(fake_dspy):
    from evomcp.backends.mutation import build_mutation_lm

    with pytest.raises(ValueError, match="unknown mutation backend"):
        build_mutation_lm({"backend": "nope", "model": "x"})


def test_claude_default(fake_dspy):
    from evomcp.backends.mutation import build_mutation_lm

    lm = build_mutation_lm(None)
    assert lm.model == "anthropic/claude-haiku-4-5"
    assert lm.kwargs["max_tokens"] == 4096


def test_claude_explicit(fake_dspy):
    from evomcp.backends.mutation import build_mutation_lm

    lm = build_mutation_lm(
        {"backend": "claude", "model": "claude-sonnet-4-5", "max_tokens": 2048}
    )
    assert lm.model == "anthropic/claude-sonnet-4-5"
    assert lm.kwargs["max_tokens"] == 2048


def test_openai(fake_dspy):
    from evomcp.backends.mutation import build_mutation_lm

    lm = build_mutation_lm({"backend": "openai", "model": "gpt-4.1-mini"})
    assert lm.model == "openai/gpt-4.1-mini"


def test_vllm_requires_base_url(fake_dspy):
    from evomcp.backends.mutation import build_mutation_lm

    with pytest.raises(ValueError, match="base_url"):
        build_mutation_lm({"backend": "vllm", "model": "Qwen/Qwen3-8B"})


def test_vllm_with_base_url(fake_dspy):
    from evomcp.backends.mutation import build_mutation_lm

    lm = build_mutation_lm(
        {
            "backend": "vllm",
            "model": "Qwen/Qwen3-8B",
            "base_url": "http://gpu-node:8000/v1",
        }
    )
    assert lm.model == "openai/Qwen/Qwen3-8B"
    assert lm.kwargs["base_url"] == "http://gpu-node:8000/v1"
    assert lm.kwargs["api_key"] == "none"


def test_openrouter_requires_api_key(fake_dspy, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from evomcp.backends.mutation import build_mutation_lm

    with pytest.raises(ValueError, match="api_key"):
        build_mutation_lm({"backend": "openrouter", "model": "anthropic/claude-haiku-4.5"})


def test_openrouter_uses_env(fake_dspy, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    from evomcp.backends.mutation import build_mutation_lm

    lm = build_mutation_lm(
        {"backend": "openrouter", "model": "anthropic/claude-haiku-4.5"}
    )
    assert lm.kwargs["api_key"] == "sk-test"
    assert "openrouter.ai" in lm.kwargs["base_url"]


def test_ssh_vllm_requires_host(fake_dspy):
    from evomcp.backends.mutation import build_mutation_lm

    with pytest.raises(ValueError, match="ssh_host"):
        build_mutation_lm({"backend": "ssh_vllm", "model": "Qwen/Qwen3-8B"})


def test_ssh_vllm_preflight_fails_unreachable(fake_dspy):
    from evomcp.backends.mutation import build_mutation_lm

    # Bogus host that cannot resolve — preflight must fail fast, not hang.
    with pytest.raises(RuntimeError, match="ssh preflight failed"):
        build_mutation_lm(
            {
                "backend": "ssh_vllm",
                "model": "Qwen/Qwen3-8B",
                "ssh_host": "nonexistent-host-does-not-resolve-evomcp",
            }
        )


def test_free_port_returns_bound_port(fake_dspy):
    from evomcp.backends.mutation import _free_port

    import socket

    port = _free_port()
    assert 1024 < port < 65536
    # The port should be re-bindable (i.e. actually free)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))
    s.close()
