"""Mutation-LM backend registry.

Replaces the hardcoded `if lm_backend == "claude" ... elif "openai"` branches
that previously lived in `gepa_runner.py:219-229` and `hybrid_runner.py:365-373`.

Supported backends (key → class):
    claude      → Anthropic API (default)
    openai      → OpenAI API
    openrouter  → OpenRouter API (OpenAI-compatible)
    codex       → Local Codex CLI (`codex exec`)
    vllm        → Any OpenAI-compatible endpoint with explicit base_url
    ssh_vllm    → vLLM behind an SSH tunnel to an arbitrary host

YAML shape:
    mutation_backend:
      backend: claude | openai | openrouter | codex | vllm | ssh_vllm
      model: <model-id>
      # backend-specific:
      base_url: <url>           # vllm, openrouter
      api_key: <key>            # optional, falls back to env
      timeout: 300              # codex
      ssh_host: <alias>         # ssh_vllm — any host from ~/.ssh/config
      remote_port: 8000         # ssh_vllm, default 8000
      local_port: 0             # ssh_vllm, 0 = pick free port
      max_tokens: 4096
"""
from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import tempfile
import threading
import time
from typing import Any, Protocol

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, "MutationBackend"] = {}


class MutationBackend(Protocol):
    name: str

    def build_lm(self, config: dict[str, Any]) -> Any:  # returns dspy.LM
        ...


def register(name: str):
    def deco(cls):
        inst = cls()
        inst.name = name
        _REGISTRY[name] = inst
        return cls
    return deco


def build_mutation_lm(config: dict[str, Any] | None) -> Any:
    """Resolve a `mutation_backend` YAML block to a configured `dspy.LM`.

    Defaults: backend=claude, model=claude-haiku-4-5, max_tokens=4096.
    Raises ValueError on unknown backend, ImportError if dspy is missing,
    RuntimeError if SSH preflight fails for ssh_vllm.
    """
    cfg = dict(config or {})
    backend = cfg.pop("backend", "claude")
    if backend not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown mutation backend {backend!r}; known: {known}")
    return _REGISTRY[backend].build_lm(cfg)


def _import_dspy():
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "mutation backends require dspy-ai. Install with: pip install dspy-ai"
        ) from exc
    return dspy


# ---------------------------------------------------------------------------
# Simple API backends
# ---------------------------------------------------------------------------

@register("claude")
class ClaudeBackend:
    def build_lm(self, cfg):
        dspy = _import_dspy()
        model = cfg.get("model", "claude-haiku-4-5")
        max_tokens = int(cfg.get("max_tokens", 4096))
        kwargs = {"max_tokens": max_tokens}
        if "api_key" in cfg:
            kwargs["api_key"] = cfg["api_key"]
        return dspy.LM(f"anthropic/{model}", **kwargs)


@register("openai")
class OpenAIBackend:
    def build_lm(self, cfg):
        dspy = _import_dspy()
        model = cfg.get("model", "gpt-4.1-mini")
        max_tokens = int(cfg.get("max_tokens", 4096))
        kwargs = {"max_tokens": max_tokens}
        if "api_key" in cfg:
            kwargs["api_key"] = cfg["api_key"]
        if "base_url" in cfg:
            kwargs["base_url"] = cfg["base_url"]
        return dspy.LM(f"openai/{model}", **kwargs)


@register("openrouter")
class OpenRouterBackend:
    def build_lm(self, cfg):
        dspy = _import_dspy()
        model = cfg.get("model", "anthropic/claude-haiku-4.5")
        max_tokens = int(cfg.get("max_tokens", 4096))
        base_url = cfg.get("base_url", "https://openrouter.ai/api/v1")
        api_key = cfg.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "openrouter backend requires api_key in config or OPENROUTER_API_KEY env var"
            )
        return dspy.LM(
            f"openai/{model}",
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
        )


# ---------------------------------------------------------------------------
# Local Codex CLI backend
# ---------------------------------------------------------------------------

class CodexCLIError(RuntimeError):
    """Raised when the local Codex CLI mutation backend cannot produce a reply."""


_CODEX_OUTPUT_LAST_MESSAGE_SUPPORT: dict[str, bool] = {}


def _stderr_tail(stderr: str | None, limit: int = 1000) -> str:
    text = (stderr or "").strip()
    return text[-limit:] if text else ""


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _compose_codex_prompt(
    prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> str:
    if messages:
        blocks = []
        for message in messages:
            role = str(message.get("role", "user")).strip() or "user"
            content = _message_content_to_text(message.get("content", ""))
            blocks.append(f"{role.upper()}:\n{content}")
        if prompt:
            blocks.append(f"USER:\n{prompt}")
        return "\n\n".join(blocks)
    return prompt or ""


def _codex_supports_output_last_message(cli: str, timeout_s: int) -> bool:
    cached = _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT.get(cli)
    if cached is not None:
        return cached
    try:
        result = subprocess.run(
            [cli, "exec", "--help"],
            capture_output=True,
            text=True,
            timeout=min(timeout_s, 10),
        )
    except (OSError, subprocess.TimeoutExpired):
        _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT[cli] = False
        return False
    combined = f"{result.stdout}\n{result.stderr}"
    supported = "--output-last-message" in combined
    _CODEX_OUTPUT_LAST_MESSAGE_SUPPORT[cli] = supported
    return supported


def _extract_codex_stdout_reply(stdout: str | None) -> str:
    text = (stdout or "").strip()
    if not text:
        return ""
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    return blocks[-1] if blocks else text


class CodexCLILM:
    """Minimal DSPy-compatible LM wrapper around `codex exec`.

    Kept dspy-free at module level (repo convention: dspy imports are lazy).
    GEPA enforces `isinstance(lm, dspy.BaseLM)`, so `CodexCLIBackend.build_lm`
    mixes the real BaseLM in dynamically at construction time.
    """

    def __init__(
        self,
        model: str = "gpt-5.5",
        timeout_s: int = 300,
        cli: str = "codex",
    ):
        self.model = model
        self.model_type = "chat"
        self.timeout_s = timeout_s
        self.cli = cli
        self.kwargs = {"timeout": timeout_s}
        self.history: list[dict[str, Any]] = []

    def copy(self, **kwargs):
        clone = type(self)(model=self.model, timeout_s=self.timeout_s, cli=self.cli)
        clone.kwargs = dict(self.kwargs)
        clone.history = []
        for key, value in kwargs.items():
            if hasattr(clone, key):
                setattr(clone, key, value)
            if key in clone.kwargs or not hasattr(clone, key):
                if value is None:
                    clone.kwargs.pop(key, None)
                else:
                    clone.kwargs[key] = value
        return clone

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[str]:
        full_prompt = _compose_codex_prompt(prompt=prompt, messages=messages)
        timeout_s = int(kwargs.get("timeout", self.timeout_s))
        reply = self._complete(full_prompt, timeout_s=timeout_s)
        self.history.append(
            {
                "prompt": prompt,
                "messages": messages,
                "kwargs": kwargs,
                "outputs": [reply],
                "model": self.model,
            }
        )
        return [reply]

    def _complete(self, prompt: str, timeout_s: int) -> str:
        use_output_file = _codex_supports_output_last_message(self.cli, timeout_s)
        last_error = "empty reply"
        last_stderr = ""

        for attempt in range(2):
            output_path = None
            try:
                cmd = [
                    self.cli,
                    "exec",
                    "-m",
                    self.model,
                    "--skip-git-repo-check",
                    "-s",
                    "read-only",
                ]
                if use_output_file:
                    handle = tempfile.NamedTemporaryFile("w+", delete=False)
                    output_path = handle.name
                    handle.close()
                    cmd.extend(["--output-last-message", output_path])
                cmd.append("-")

                result = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                last_stderr = result.stderr
                if result.returncode != 0:
                    last_error = f"exit code {result.returncode}"
                    continue

                if output_path:
                    with open(output_path, encoding="utf-8") as fh:
                        reply = fh.read().strip()
                else:
                    reply = _extract_codex_stdout_reply(result.stdout)

                if reply:
                    return reply
                last_error = "empty reply"
            except subprocess.TimeoutExpired as exc:
                last_error = f"timeout after {timeout_s}s"
                last_stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            except OSError as exc:
                last_error = str(exc)
                last_stderr = ""
            finally:
                if output_path:
                    try:
                        os.unlink(output_path)
                    except FileNotFoundError:
                        pass

            if attempt == 0:
                log.warning("codex CLI mutation attempt failed: %s", last_error)

        tail = _stderr_tail(last_stderr)
        detail = f"; stderr tail: {tail}" if tail else ""
        raise CodexCLIError(
            f"codex CLI mutation failed after 2 attempts: {last_error}{detail}"
        )


@register("codex")
class CodexCLIBackend:
    def build_lm(self, cfg):
        dspy = _import_dspy()
        model = cfg.get("model", "gpt-5.5")
        timeout_s = int(cfg.get("timeout", 300))
        cli = cfg.get("cli", "codex")
        # GEPA проверяет isinstance(lm, dspy.BaseLM); подмешиваем BaseLM в
        # рантайме, не теряя dspy-free импорт модуля. MRO: CodexCLILM первым,
        # его __init__/__call__ перекрывают BaseLM (CLI — это транспорт).
        cls = type("CodexCLIBaseLM", (CodexCLILM, dspy.BaseLM), {})
        return cls(model=model, timeout_s=timeout_s, cli=cli)


@register("vllm")
class VLLMBackend:
    """OpenAI-compatible endpoint (vLLM, llama.cpp, text-generation-inference, ...)."""

    def build_lm(self, cfg):
        dspy = _import_dspy()
        model = cfg.get("model")
        if not model:
            raise ValueError("vllm backend requires `model`")
        base_url = cfg.get("base_url")
        if not base_url:
            raise ValueError("vllm backend requires `base_url`")
        max_tokens = int(cfg.get("max_tokens", 4096))
        api_key = cfg.get("api_key", "none")
        return dspy.LM(
            f"openai/{model}",
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
        )


# ---------------------------------------------------------------------------
# SSH tunnel helper + ssh_vllm backend
# ---------------------------------------------------------------------------

_tunnel_lock = threading.Lock()
_TUNNELS: dict[tuple[str, int], "_SSHTunnel"] = {}


def _free_port() -> int:
    """Ask the OS for an unused TCP port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _wait_port_open(host: str, port: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.1)
    return False


class _SSHTunnel:
    """Background `ssh -L` tunnel, torn down on process exit."""

    def __init__(self, ssh_host: str, remote_port: int, local_port: int):
        self.ssh_host = ssh_host
        self.remote_port = remote_port
        self.local_port = local_port
        self.proc: subprocess.Popen | None = None

    def open(self) -> None:
        # -N: no remote command, -T: no TTY, -o ExitOnForwardFailure=yes:
        # fail fast if the forward cannot be established.
        cmd = [
            "ssh",
            "-N",
            "-T",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-L", f"{self.local_port}:localhost:{self.remote_port}",
            self.ssh_host,
        ]
        log.info("opening ssh tunnel: %s", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if not _wait_port_open("127.0.0.1", self.local_port, timeout=8.0):
            self.close()
            raise RuntimeError(
                f"ssh tunnel to {self.ssh_host}:{self.remote_port} "
                f"did not become reachable at localhost:{self.local_port} within 8s"
            )

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def _ensure_tunnel(ssh_host: str, remote_port: int, local_port: int | None) -> int:
    """Return the local port bound to a live tunnel. Cached per (host, remote_port)."""
    key = (ssh_host, remote_port)
    with _tunnel_lock:
        tun = _TUNNELS.get(key)
        if tun is not None and tun.proc and tun.proc.poll() is None:
            return tun.local_port
        chosen = local_port or _free_port()
        tun = _SSHTunnel(ssh_host, remote_port, chosen)
        tun.open()
        _TUNNELS[key] = tun
        return tun.local_port


def _preflight_ssh(ssh_host: str, timeout_s: int = 5) -> None:
    """Raise RuntimeError unless `ssh <host> true` succeeds non-interactively."""
    result = subprocess.run(
        [
            "ssh",
            "-o", f"ConnectTimeout={timeout_s}",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            ssh_host,
            "true",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ssh preflight failed for {ssh_host!r}: "
            f"rc={result.returncode} stderr={result.stderr.strip()[:200]!r}"
        )


@atexit.register
def _teardown_tunnels() -> None:
    for tun in list(_TUNNELS.values()):
        try:
            tun.close()
        except Exception as exc:
            log.warning("tunnel teardown failed: %s", exc)
    _TUNNELS.clear()


@register("ssh_vllm")
class SSHVLLMBackend:
    """vLLM behind an SSH tunnel. Auto-manages tunnel lifecycle."""

    def build_lm(self, cfg):
        dspy = _import_dspy()
        model = cfg.get("model")
        if not model:
            raise ValueError("ssh_vllm backend requires `model`")
        ssh_host = cfg.get("ssh_host")
        if not ssh_host:
            raise ValueError("ssh_vllm backend requires `ssh_host`")
        remote_port = int(cfg.get("remote_port", 8000))
        local_port_cfg = cfg.get("local_port") or None
        max_tokens = int(cfg.get("max_tokens", 4096))
        api_key = cfg.get("api_key", "none")

        _preflight_ssh(ssh_host)
        local_port = _ensure_tunnel(ssh_host, remote_port, local_port_cfg)
        base_url = f"http://127.0.0.1:{local_port}/v1"
        log.info("ssh_vllm: %s → %s (model=%s)", ssh_host, base_url, model)
        return dspy.LM(
            f"openai/{model}",
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
        )
