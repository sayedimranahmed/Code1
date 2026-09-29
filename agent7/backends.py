"""Pluggable model backends. A backend is anything with
`complete(system: str, messages: list[dict], effort: str) -> str`.

Only the Anthropic backend talks to a real model. The other two exist to
test the harness without spending tokens or needing a key.
"""
import os
import threading

from .tasks import TASKS


class BackendError(RuntimeError):
    pass


class AnthropicBackend:
    """Claude via the official SDK (`pip install anthropic`).

    Credentials resolve the SDK's usual way (ANTHROPIC_API_KEY, or an
    `ant auth login` profile). Nothing is hardcoded. Current models reject
    `temperature`, so the sampling knob the optimizer can turn is `effort`.
    Refusals fall back server-side to another model (set AGENT7_NO_FALLBACK=1
    to turn that off).
    """

    def __init__(self, model=None, max_tokens=16000):
        try:
            import anthropic
        except ImportError as e:
            raise BackendError("pip install anthropic") from e
        self._anthropic = anthropic
        # Explicit base URL: hosted sandboxes set ANTHROPIC_BASE_URL for their own
        # traffic, and the SDK would silently follow it.
        base = os.environ.get("AGENT7_BASE_URL", "https://api.anthropic.com")
        self.client = anthropic.Anthropic(base_url=base, max_retries=5)
        self.model = model or os.environ.get("AGENT7_MODEL", "claude-opus-5-5")
        self.max_tokens = max_tokens
        self.fallback = os.environ.get("AGENT7_NO_FALLBACK") != "1"
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "refusals": 0}
        self._lock = threading.Lock()

    def complete(self, system, messages, effort="medium"):
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages,
                  output_config={"effort": effort})
        try:
            if self.fallback:
                r = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
            else:
                r = self.client.messages.create(**kw)
        except self._anthropic.APIError as e:
            raise BackendError(str(e)) from e
        with self._lock:
            self.usage["input_tokens"] += r.usage.input_tokens
            self.usage["output_tokens"] += r.usage.output_tokens
            self.usage["calls"] += 1
            self.usage["refusals"] += r.stop_reason == "refusal"
        if r.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in r.content if b.type == "text")


class OracleBackend:
    """Answers every task with its reference solution. Used by `validate` to
    prove the pipeline end to end: prompt -> extraction -> sandbox -> score
    must give 100%. It cannot answer optimizer prompts."""

    def complete(self, system, messages, effort="medium"):
        text = messages[0]["content"]
        for t in TASKS:
            if f"TASK_ID: {t.id}\n" in text:
                return "```python\n" + t.reference + "```"
        raise BackendError("oracle only answers task prompts")


class ScriptedBackend:
    """Returns canned responses in order, or calls a function. For unit tests."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def complete(self, system, messages, effort="medium"):
        self.calls.append((system, messages, effort))
        if callable(self.responses):
            return self.responses(system, messages)
        return self.responses.pop(0)


def make_backend(name, model=None):
    if name == "anthropic":
        return AnthropicBackend(model=model)
    if name == "oracle":
        return OracleBackend()
    raise BackendError(f"unknown backend {name!r}")
