"""Groq transport: retries, backoff and token accounting.

Deliberately contains no prompt text -- that lives in src/agent/prompts.py.
This layer knows how to talk to the API and nothing about churn.

Rate limits are part of the design problem, not an obstacle to route around
(brief, section 3). Every call is counted so the cost of the agent loop is a
measured number rather than a claim.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Model precedence, highest first:
#   1. the `model=` argument
#   2. GROQ_MODEL in the environment / .env      <- the normal way to set this
#   3. GROQ_MODEL_FALLBACKS (comma-separated), or the built-in list below
#
# The built-in list exists only so a fresh clone with no GROQ_MODEL still runs,
# and so a configured model the account cannot access degrades to a logged
# warning instead of failing every request. The historically recommended
# llama-3.3-70b-versatile is NOT on current free accounts, which is why
# availability is verified rather than assumed.
BUILTIN_FALLBACKS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]

DEFAULT_MODEL = BUILTIN_FALLBACKS[0]


def _fallback_models() -> list[str]:
    raw = os.environ.get("GROQ_MODEL_FALLBACKS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(BUILTIN_FALLBACKS)


# Backwards-compatible alias; prefer _fallback_models() so the env is respected.
PREFERRED_MODELS = BUILTIN_FALLBACKS
MAX_RETRIES = 4
BASE_BACKOFF = 2.0


class LLMUnavailable(RuntimeError):
    """No usable API key, or the provider is unreachable after retries."""


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env loader.

    python-dotenv is in requirements, but this keeps the module importable in a
    notebook or a container where the file may be absent. Existing environment
    variables always win, so Docker's -e and Streamlit's secrets take precedence.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _is_json_validate_failure(exc: Exception) -> bool:
    """The provider rejected the completion because it was not valid JSON."""
    return "json_validate_failed" in str(exc)


def _status_code(exc: Exception) -> int | None:
    """Dig the HTTP status out of a provider exception.

    The SDK does not expose it uniformly across error types, and defaulting an
    unknown status to "retryable" means a hard 400 gets retried until the budget
    is gone. Falls back to parsing the "Error code: NNN" text the SDK embeds.
    """
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    match = re.search(r"Error code:\s*(\d{3})", str(exc))
    return int(match.group(1)) if match else None


@dataclass
class Usage:
    """Per-session token accounting, reported in the README as evidence."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    rate_limited: int = 0
    seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int, seconds: float) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.seconds += seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "retries": self.retries,
            "rate_limited": self.rate_limited,
            "seconds": round(self.seconds, 2),
            "avg_tokens_per_call": round(self.total_tokens / self.calls, 1) if self.calls else 0,
        }


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    seconds: float = 0.0

    def json(self) -> dict[str, Any]:
        """Parse the response as JSON, tolerating fenced or prose-wrapped output.

        Free-tier models wrap JSON in ```json fences or a sentence of preamble
        often enough that failing on it would be a self-inflicted error.
        """
        text = self.text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise


class LLMClient:
    """Thin Groq wrapper. One instance per session so Usage accumulates."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 auto_select_model: bool = True) -> None:
        load_env()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "").strip()
        self.usage = Usage()
        self._client = None
        env_model = (os.environ.get("GROQ_MODEL") or "").strip()
        if model:
            self.model, self.model_source = model.strip(), "argument"
        elif env_model:
            self.model, self.model_source = env_model, "GROQ_MODEL"
        else:
            self.model, self.model_source = DEFAULT_MODEL, "built-in default"

        if not self.api_key:
            self.available = False
            self.reason = (
                "GROQ_API_KEY is not set. Copy .env.example to .env and add a free key "
                "from https://console.groq.com. The Explore and What-if tabs work without it."
            )
            return

        try:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
            self.available = True
            self.reason = ""
        except Exception as exc:
            self.available = False
            self.reason = f"Could not initialise the Groq client: {exc}"
            return

        if auto_select_model:
            self._ensure_model_available()

    def _ensure_model_available(self) -> None:
        """Fall back to a model this account can actually use.

        A configured model that the account cannot access fails on every request
        with an opaque error. One cheap lookup at startup turns that into a
        logged fallback instead of a broken session.
        """
        try:
            ids = {m.id for m in self._client.models.list().data}
        except Exception as exc:
            logger.warning("Could not list Groq models (%s); using %s as configured.", exc, self.model)
            return

        if self.model in ids:
            logger.info("Using model %r (from %s).", self.model, self.model_source)
            return
        for candidate in _fallback_models():
            if candidate in ids:
                logger.warning(
                    "Model %r (from %s) is not available on this account; falling back to %r. "
                    "Set GROQ_MODEL in .env to one of: %s",
                    self.model, self.model_source, candidate, ", ".join(sorted(ids)),
                )
                self.model, self.model_source = candidate, "fallback"
                return
        logger.error("No usable chat model found. Available on this account: %s", sorted(ids))

    # ------------------------------------------------------------------
    def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """One chat completion, with backoff on rate limits and transient errors.

        Raises LLMUnavailable rather than returning a sentinel: the caller (the
        agent loop) needs to distinguish "the model is down" from "the model
        answered badly", and those demand different handling.
        """
        if not self.available:
            raise LLMUnavailable(self.reason)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            # gpt-oss models spend tokens reasoning before emitting content. Too
            # small a budget yields an empty completion, which the API rejects as
            # json_validate_failed -- an error that looks like a prompt problem
            # but is really a truncation problem.
            kwargs["max_tokens"] = max(max_tokens, 1024)

        last_error: Exception | None = None
        attempts = 0
        for attempt in range(MAX_RETRIES):
            attempts = attempt + 1
            try:
                started = time.time()
                response = self._client.chat.completions.create(**kwargs)
                elapsed = time.time() - started
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                self.usage.add(prompt_tokens, completion_tokens, elapsed)
                return LLMResponse(
                    text=response.choices[0].message.content or "",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model=self.model,
                    seconds=elapsed,
                )
            except Exception as exc:
                last_error = exc
                status = _status_code(exc)
                is_rate_limit = status == 429
                # 4xx other than 429 means the request itself is wrong -- retrying
                # it just burns rate-limit budget to get the same error back.
                retryable = is_rate_limit or status is None or status >= 500

                # A model that answers a greeting conversationally cannot satisfy
                # response_format=json_object, and the API rejects the completion
                # with 400 json_validate_failed. Dropping strict JSON and parsing
                # leniently recovers the turn; the brief explicitly allows prompted
                # JSON as a fallback.
                if json_mode and _is_json_validate_failure(exc) and "response_format" in kwargs:
                    logger.warning("Strict JSON rejected by the model; retrying without it.")
                    kwargs.pop("response_format")
                    continue

                if not retryable or attempt == MAX_RETRIES - 1:
                    break

                if is_rate_limit:
                    self.usage.rate_limited += 1
                self.usage.retries += 1
                delay = self._retry_delay(exc, attempt)
                logger.warning("Groq call failed (%s); retrying in %.1fs", type(exc).__name__, delay)
                time.sleep(delay)

        plural = "attempt" if attempts == 1 else "attempts"
        raise LLMUnavailable(f"Groq request failed after {attempts} {plural}: {last_error}")

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        """Honour Retry-After when the provider sends one; otherwise back off.

        Jitter matters: without it, concurrent Streamlit sessions retry in
        lockstep and reproduce the burst that caused the limit.
        """
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {}) or {}
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
        if retry_after is not None:
            return min(retry_after, 30.0)
        return min(BASE_BACKOFF * (2 ** attempt), 30.0) + random.uniform(0, 0.5)


_shared: LLMClient | None = None


def get_client() -> LLMClient:
    """Process-wide client, so token accounting spans the whole session."""
    global _shared
    if _shared is None:
        _shared = LLMClient()
    return _shared
