"""Per-run setting overrides, so one process can run experiments fairly.

A/B experiments ("same requests, model A vs model B", "retry limit 2 vs 5",
"memory on vs off") need those settings to differ BETWEEN runs that execute at
the same time. The globals in app.config can't do that, so a run's overrides
live in a ContextVar: set at the start of run_pipeline (inside that run's own
asyncio task) and automatically visible in everything the run does, including
worker threads started with asyncio.to_thread, without leaking into any other
concurrent run.

Supported overrides (all optional):
    model            Ollama model name for every generate call of the run
    max_qa_attempts  QA attempts per task before escalating (1-10)
    use_memory       False = long-term memory is neither read nor written
                     (task memory AND failure patterns)
    failure_recovery False = a retry sees only the raw QA error text (the pre-Phase-12
                     behaviour): no parsed diagnosis, no line-numbered previous code,
                     no "already tried" list, no similar-fix hints, no LLM refinement.
                     Failures are still analysed for METRICS (repeated_errors etc.), so
                     both arms are measured the same way; only the Developer's prompt
                     differs. Default True.
    temperature      Ollama sampling temperature (0-2) for every generate call of the run
    seed             Ollama sampling seed (integer >= 0)
    num_ctx          Ollama context window in tokens (512-131072)

Everything else (communication mode, framework, interactivity) is already a
per-run parameter of run_pipeline.
"""
import re
from contextvars import ContextVar, Token
from typing import Any, Optional

from app.config import settings

ALLOWED = ("model", "max_qa_attempts", "use_memory", "failure_recovery", "temperature", "seed", "num_ctx")
_MODEL_RE = re.compile(r"^[\w.:/@+-]{1,100}$")
_overrides: ContextVar[dict] = ContextVar("aiopsforge_run_overrides", default={})


def validate(overrides: Optional[dict]) -> dict:
    """Returns a cleaned copy of `overrides` or raises ValueError. Unknown keys
    are rejected (a typo'd experiment factor must not silently do nothing)."""
    out: dict[str, Any] = {}
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k not in ALLOWED:
            raise ValueError(f"unknown run override {k!r}; supported: {', '.join(ALLOWED)}")
        if k == "model":
            if not isinstance(v, str) or not _MODEL_RE.match(v):
                raise ValueError("model must be an Ollama model name such as 'qwen2.5-coder:3b'")
            out[k] = v
        elif k == "max_qa_attempts":
            if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 10:
                raise ValueError("max_qa_attempts must be an integer from 1 to 10")
            out[k] = v
        elif k == "temperature":
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= float(v) <= 2:
                raise ValueError("temperature must be a number from 0 to 2")
            out[k] = float(v)
        elif k == "seed":
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 2**31 - 1:
                raise ValueError("seed must be an integer from 0 to 2147483647")
            out[k] = v
        elif k == "num_ctx":
            if isinstance(v, bool) or not isinstance(v, int) or not 512 <= v <= 131072:
                raise ValueError("num_ctx must be an integer from 512 to 131072")
            out[k] = v
        else:  # use_memory, failure_recovery
            if not isinstance(v, bool):
                raise ValueError(f"{k} must be true or false")
            out[k] = v
    return out


def push(overrides: Optional[dict]) -> Token:
    """Activates `overrides` for the current context. Pair with pop()."""
    return _overrides.set(validate(overrides))


def pop(token: Token) -> None:
    try:
        _overrides.reset(token)
    except ValueError:  # reset from a different context: the run's own task ended anyway
        pass


def current() -> dict:
    return dict(_overrides.get())


def model() -> str:
    return _overrides.get().get("model") or settings.ollama_model


def max_qa_attempts() -> int:
    return _overrides.get().get("max_qa_attempts") or settings.max_qa_attempts


def memory_enabled() -> bool:
    return _overrides.get().get("use_memory", True)


def failure_recovery_enabled() -> bool:
    return _overrides.get().get("failure_recovery", True)


def llm_options() -> dict:
    """Ollama `options` for a generate call: the run's override, else the
    configured default, else nothing (Ollama's own default). Only keys that are
    actually set are returned, so an unconfigured install sends no `options`."""
    ov = _overrides.get()
    out: dict[str, Any] = {}
    for key, default in (("temperature", settings.ollama_temperature),
                         ("seed", settings.ollama_seed),
                         ("num_ctx", settings.ollama_num_ctx)):
        val = ov.get(key, default)
        if val is not None:
            out[key] = val
    return out
