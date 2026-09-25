"""Planner Agent (Week 2 / Days 6-10).

Takes a plain-English request and turns it into an ordered list of small,
concrete coding tasks the Developer Agent can tackle one at a time. This is
the first of the three real agents in the pipeline — everything before this
(Phase 1, and the MCP layer in Week 3) was infrastructure for this to run on.
"""
from app.llm import generate, generate_json, LLMOutputError
from app.schemas import Task
from app.webapp_detect import is_webapp_request

PLANNER_SYSTEM_PROMPT = """You are the Planner Agent in an automated software \
development pipeline. Given a user's request, break it into a short, ordered \
list of small, concrete coding tasks. Each task should be independently \
implementable in one file and testable.

Respond with ONLY a JSON object of this exact shape, nothing else:
{"tasks": ["first task description", "second task description", ...]}

Keep the list short: 1-4 tasks for a simple request, up to 6-8 for a more \
involved one. Do not include setup/deployment tasks (e.g. "set up the \
repository") — only tasks that produce code and/or tests."""

WEBAPP_SUMMARY_SYSTEM_PROMPT = """You are the Planner Agent in an automated \
software development pipeline. Given a user's request for a website or web \
app, write a short summary (2-4 sentences) of what to build — its purpose, \
its pages/routes, and its key features — the way a project lead would hand \
a task off to a developer. Paraphrase and compress; do not just repeat the \
request back verbatim. Respond with ONLY the summary text, nothing else."""


def _summarize_webapp_request(request: str) -> str:
    """Produces the Planner's SUMMARY of a website request — this is what
    becomes the webapp_scaffold Task's description, distinct from the raw
    request text itself. This distinction matters: it's what makes the
    communication-mode ablation (Phase 6) meaningful for webapp tasks too —
    in "message_passing" mode the Scaffolder sees only this summary; in
    "blackboard" mode it also sees the original raw request. Without a real
    summarization step, task.description would just be the raw request
    again and there'd be nothing for either mode to lose.

    Falls back to the raw request itself if the summarization call fails —
    a slightly-too-literal task description is harmless; a crash isn't.
    """
    try:
        summary = generate(
            f"User request:\n{request}\n\nWrite the summary described above.",
            system=WEBAPP_SUMMARY_SYSTEM_PROMPT,
        )
        summary = (summary or "").strip()
        return summary if summary else request.strip()
    except Exception:  # noqa: BLE001 — any LLM/network failure, never crash the Planner
        return request.strip()


def plan(request: str) -> list[Task]:
    """Calls the LLM to break `request` into an ordered Task list.

    Phase 7: if `request` reads as a website/web-app build (see
    app.webapp_detect), skips the normal per-function decomposition
    entirely and returns a single "webapp_scaffold" task instead — a
    website isn't naturally a sequence of independent, separately-testable
    functions the way the rest of this pipeline assumes, so treating
    scaffolding as one cohesive unit (handled by app.agents.scaffolder)
    is a better fit than forcing it through the normal task list.

    Falls back to a single catch-all task if the model's output can't be
    parsed even after llm.generate_json's internal retries — a plan with one
    broad task is more useful to the rest of the pipeline than a hard crash.
    """
    if is_webapp_request(request):
        summary = _summarize_webapp_request(request)
        return [Task(order=1, description=summary, type="webapp_scaffold")]

    prompt = f"User request:\n{request}\n\nRespond with the JSON object described above."
    try:
        parsed = generate_json(prompt, system=PLANNER_SYSTEM_PROMPT)
        descriptions = parsed.get("tasks", [])
        if not isinstance(descriptions, list) or not descriptions:
            raise ValueError("'tasks' was empty or not a list")
    except (LLMOutputError, KeyError, ValueError):
        descriptions = [request]

    result = []
    for i, desc in enumerate(descriptions, start=1):
        # Handle both strings and dicts (LLM may return either)
        if isinstance(desc, dict):
            desc = desc.get("description") or desc.get("task") or str(desc)
        desc = str(desc).strip() if desc else ""
        if desc:
            result.append(Task(order=i, description=desc))
    
    return result if result else [Task(order=1, description=request)]
