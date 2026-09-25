"""Thin wrapper around the Ollama HTTP API, shared by all three agents.

Kept in one place so retry/timeout/JSON-parsing behaviour is consistent
across Planner, Developer, and QA instead of each agent reinventing it.

Every call is wrapped with Langfuse's @observe decorator (Week 10) for
prompt/latency/token tracing.

Bug fix (found via production logs): the docstring here used to claim that
leaving LANGFUSE_PUBLIC_KEY unset makes @observe "a harmless no-op." That
was wrong — the decorator still tries to export spans to LANGFUSE_HOST
(which docker-compose defaults to Langfuse's real cloud host even with no
key configured), producing a 401 Unauthorized on every single LLM call and
flooding the logs, which made the actual errors below much harder to spot
during debugging. Langfuse's own documented fix is the LANGFUSE_TRACING_ENABLED
env var — set it to "false" here, before `observe`/`get_client` are
imported, whenever no key is configured, so tracing is genuinely disabled
rather than silently failing on every call.
"""
import json
import os
import re
import time

import requests

from app.config import settings

if not settings.langfuse_public_key:
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")

from langfuse import observe, get_client


class LLMOutputError(Exception):
    """Raised when the model's response can't be parsed the way the caller needed."""


@observe(name="ollama-generate", as_type="generation")
def generate(prompt: str, system: str | None = None, timeout: int = 600) -> str:
    """Sends a single prompt to Ollama and returns the raw text response."""
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        # Reuse the already-loaded model across the many back-to-back calls
        # one pipeline run makes (planner, then developer per task/attempt,
        # then failure_analyzer on retry) instead of Ollama's default
        # 5-minute keep_alive, which is easy to exceed on a slow local
        # model and forces a full reload mid-run.
        "keep_alive": "30m",
    }
    if system:
        payload["system"] = system

    t0 = time.time()
    response = requests.post(
        f"{settings.ollama_base_url}/api/generate", json=payload, timeout=timeout
    )
    response.raise_for_status()
    body = response.json()
    text_out = body.get("response", "").strip()

    if settings.langfuse_public_key:
        try:
            client = get_client()
            client.update_current_generation(
                model=settings.ollama_model,
                input=prompt,
                output=text_out,
                usage_details={
                    "input": body.get("prompt_eval_count"),
                    "output": body.get("eval_count"),
                },
                metadata={"duration_s": round(time.time() - t0, 2)},
            )
        except Exception:  # noqa: BLE001 — tracing must never break generation
            pass

    return text_out


@observe(name="ollama-generate-json")
def generate_json(prompt: str, system: str | None = None, retries: int = 2) -> dict:
    """Sends a prompt that asks for JSON back, and parses it — with a couple
    of retries and some tolerance for models that wrap JSON in prose or
    markdown code fences, since local models don't always follow
    instructions as strictly as hosted ones.
    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        raw = generate(prompt, system=system)
        try:
            return _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            prompt = (
                f"{prompt}\n\nYour previous response could not be parsed as JSON "
                f"(error: {exc}). Respond with ONLY valid JSON, no prose, no "
                f"markdown code fences."
            )
    raise LLMOutputError(f"Model did not return valid JSON after {retries + 1} attempts: {last_error}")


def _extract_json(raw: str) -> dict:
    """Pulls a JSON object out of a raw LLM response, tolerating ```json
    fences or leading/trailing prose around the object.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start : end + 1])

    return json.loads(raw)
