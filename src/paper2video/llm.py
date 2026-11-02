from __future__ import annotations
import json
import os
import subprocess
import time
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str, system: str | None = None) -> str: ...
    def complete_json(self, prompt: str, system: str | None = None) -> dict: ...


class EmptyCompletionError(RuntimeError):
    """Model returned no text blocks (e.g. thinking consumed the whole budget)."""


class FakeLLMClient:
    """Test double. Returns queued responses in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if not self._responses:
            raise RuntimeError("FakeLLMClient ran out of responses")
        return self._responses.pop(0)

    def complete_json(self, prompt: str, system: str | None = None) -> dict:
        raw = self.complete(prompt, system)
        return json.loads(raw)


class AnthropicClient:
    """Production LLM client backed by Anthropic API."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic  # lazy
        self._client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.model = model or os.environ.get("PAPER2VIDEO_MODEL", "claude-sonnet-4-6")

    def complete(self, prompt: str, system: str | None = None) -> str:
        import anthropic  # lazy
        kwargs = {
            "model": self.model,
            "max_tokens": int(os.environ.get("PAPER2VIDEO_MAX_TOKENS", "4096")),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        # Retry transient errors (connection, timeout, 5xx, 529 overloaded).
        # Capped backoff so we don't burn 20 minutes on a single scene.
        retryable = (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
            EmptyCompletionError,
        )
        backoff_schedule = [3, 8, 20, 45, 90]  # ~166s total, then raise
        last_exc: Exception | None = None
        for attempt, wait in enumerate([*backoff_schedule, None]):
            try:
                msg = self._client.messages.create(**kwargs)
                # GLM-style responses may lead with thinking blocks; keep all
                # text blocks and skip empty payloads.
                parts = [
                    getattr(b, "text", "") for b in msg.content
                    if getattr(b, "type", "") == "text"
                ]
                if not parts:
                    raise EmptyCompletionError("response contained no text blocks")
                return "".join(parts)
            except retryable as e:
                last_exc = e
            except anthropic.APIStatusError as e:
                if e.status_code not in (429, 502, 503, 504, 529):
                    raise
                last_exc = e
            if wait is None:
                break
            time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def complete_json(self, prompt: str, system: str | None = None) -> dict:
        sys_prompt = (system or "") + "\n\nRespond with valid JSON only. No prose, no code fences."
        err: json.JSONDecodeError | None = None
        for attempt in range(3):
            p = prompt
            if attempt > 0 and err is not None:
                p = (
                    prompt
                    + f"\n\nNOTE: your previous reply was not valid JSON "
                    f"(error near position {err.pos}: {err.msg}). "
                    "Resend the COMPLETE corrected JSON. Escape every double quote "
                    "inside string values (\\\") or replace inner double quotes with single quotes."
                )
            raw = self.complete(p, sys_prompt.strip())
            try:
                return _parse_json_payload(raw)
            except json.JSONDecodeError as e:
                err = e
        assert err is not None
        raise err


def _parse_json_payload(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Salvage: some models wrap JSON in prose or truncate trailing fences.
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


class CodexCLIClient:
    """LLM client backed by `codex exec` (uses local Codex CLI auth, e.g. gpt-6-luna)."""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("PAPER2VIDEO_MODEL", "gpt-6-luna")

    def complete(self, prompt: str, system: str | None = None) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        cmd = [
            "codex", "exec",
            "-s", "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--json",
            "-m", self.model,
            full_prompt,
        ]
        # Retry transient failures (non-zero exit, missing agent_message, timeout).
        backoff_schedule = [5, 15, 45]
        last_exc: Exception | None = None
        for wait in [*backoff_schedule, None]:
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=900,
                    stdin=subprocess.DEVNULL,
                )
                if proc.returncode == 0:
                    text = _last_agent_message(proc.stdout)
                    if text is not None:
                        return text
                    last_exc = RuntimeError(
                        "codex exec produced no agent_message; stderr tail: "
                        + proc.stderr[-500:]
                    )
                else:
                    last_exc = RuntimeError(
                        f"codex exec exited {proc.returncode}; stderr tail: {proc.stderr[-500:]}"
                    )
            except subprocess.TimeoutExpired as e:
                last_exc = e
            if wait is None:
                break
            time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def complete_json(self, prompt: str, system: str | None = None) -> dict:
        sys_prompt = (system or "") + "\n\nRespond with valid JSON only. No prose, no code fences."
        raw = self.complete(prompt, sys_prompt.strip())
        return _parse_json_payload(raw)


def _last_agent_message(stdout: str) -> str | None:
    """Extract the final agent_message text from `codex exec --json` JSONL output."""
    text: str | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "item.completed":
            item = ev.get("item") or {}
            if item.get("type") == "agent_message":
                text = item.get("text")
    return text
