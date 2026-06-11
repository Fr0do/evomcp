"""Unit tests for evomcp.backends.mutation."""
from __future__ import annotations

import os
import subprocess
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


def test_codex_backend_uses_output_file_and_stdin(monkeypatch):
    from evomcp.backends.mutation import (
        _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT,
        build_mutation_lm,
    )

    _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT.clear()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd == ["codex", "exec", "--help"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="  -o, --output-last-message <FILE>", stderr=""
            )

        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("pong\n")
        return subprocess.CompletedProcess(
            cmd, 0, stdout="banner", stderr="wss reconnect"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    lm = build_mutation_lm({"backend": "codex", "model": "gpt-5.5", "timeout": 12})
    out = lm(prompt="Reply pong")

    assert out == ["pong"]
    exec_cmd, exec_kwargs = calls[1]
    assert exec_cmd[:4] == ["codex", "exec", "-m", "gpt-5.5"]
    assert "--skip-git-repo-check" in exec_cmd
    assert exec_cmd[exec_cmd.index("-s") + 1] == "read-only"
    assert exec_cmd[-1] == "-"
    assert exec_kwargs["input"] == "Reply pong"
    assert exec_kwargs["timeout"] == 12


def test_codex_backend_composes_chat_messages(monkeypatch):
    from evomcp.backends.mutation import (
        _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT,
        build_mutation_lm,
    )

    _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT.clear()
    prompts = []

    def fake_run(cmd, **kwargs):
        if cmd == ["codex", "exec", "--help"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="--output-last-message", stderr=""
            )
        prompts.append(kwargs["input"])
        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("ok")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    lm = build_mutation_lm({"backend": "codex"})
    lm(
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say ok."},
        ]
    )

    assert prompts == ["SYSTEM:\nBe concise.\n\nUSER:\nSay ok."]


def test_codex_backend_retries_nonzero_then_succeeds(monkeypatch):
    from evomcp.backends.mutation import (
        _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT,
        build_mutation_lm,
    )

    _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT.clear()
    exec_calls = 0

    def fake_run(cmd, **kwargs):
        nonlocal exec_calls
        if cmd == ["codex", "exec", "--help"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="--output-last-message", stderr=""
            )

        exec_calls += 1
        output_path = cmd[cmd.index("--output-last-message") + 1]
        if exec_calls == 1:
            return subprocess.CompletedProcess(
                cmd, 1, stdout="", stderr="temporary failure"
            )
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("retry-ok")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    lm = build_mutation_lm({"backend": "codex"})
    assert lm(prompt="test") == ["retry-ok"]
    assert exec_calls == 2


def test_codex_backend_raises_after_empty_retries(monkeypatch):
    from evomcp.backends.mutation import (
        CodexCLIError,
        _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT,
        build_mutation_lm,
    )

    _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT.clear()

    def fake_run(cmd, **kwargs):
        if cmd == ["codex", "exec", "--help"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="--output-last-message", stderr=""
            )

        output_path = cmd[cmd.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="last stderr")

    monkeypatch.setattr(subprocess, "run", fake_run)

    lm = build_mutation_lm({"backend": "codex"})
    with pytest.raises(CodexCLIError, match="empty reply.*last stderr"):
        lm(prompt="test")


def test_codex_backend_falls_back_to_stdout(monkeypatch):
    from evomcp.backends.mutation import (
        _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT,
        build_mutation_lm,
    )

    _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT.clear()

    def fake_run(cmd, **kwargs):
        if cmd == ["codex", "exec", "--help"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="no output flag", stderr="")
        assert "--output-last-message" not in cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout="banner\n\nfinal answer", stderr="ignored warning"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    lm = build_mutation_lm({"backend": "codex"})
    assert lm(prompt="test") == ["final answer"]


@pytest.mark.skipif(
    os.environ.get("EVOMCP_CODEX_LIVE") != "1",
    reason="set EVOMCP_CODEX_LIVE=1 to run the real Codex CLI",
)
def test_codex_backend_live_smoke():
    from evomcp.backends.mutation import build_mutation_lm

    lm = build_mutation_lm({"backend": "codex", "model": "gpt-5.5"})
    reply = lm(prompt="Reply with exactly one word: ping")[0]
    assert reply.strip().lower() == "ping"


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

    try:
        port = _free_port()
    except PermissionError as exc:
        pytest.skip(f"socket bind denied by local sandbox: {exc}")
    assert 1024 < port < 65536
    # The port should be re-bindable (i.e. actually free)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))
    s.close()
