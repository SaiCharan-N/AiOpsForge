"""Developer Agent (Week 4 / Days 16-20, retry logic added Week 5).

Takes ONE task from the Planner's list and writes code for it, then uses the
MCP write_file tool to persist it into the project's isolated workspace —
this is the "dynamic tool binding" piece: the Developer never imports a
filesystem function directly, it discovers and calls write_file over MCP.

On a retry (attempt_count > 0), the previous QA failure is fed back into the
prompt so the model is fixing a known, specific problem rather than
regenerating blind.
"""
import re

from app.llm import generate
from app.mcp_client import call_tool
from app.schemas import Task, TestResult

DEVELOPER_SYSTEM_PROMPT = """You are the Developer Agent in an automated \
software development pipeline. Given one coding task, write a single, \
complete Python file that implements it.

Rules:
- Respond with ONLY a python code block (```python ... ```) and nothing else.
- The file must be runnable/importable on its own.
- If the task is about writing tests, use pytest-style `def test_...():` \
functions and `assert` statements — do not use unittest.TestCase.
- Keep it simple and correct over clever."""


def _suggest_filename(task: Task) -> str:
    """Derives a reasonable snake_case filename from the task description.
    Test-sounding tasks get a test_*.py name so pytest picks them up.
    """
    words = re.findall(r"[a-zA-Z0-9]+", task.description.lower())[:5]
    slug = "_".join(words) or f"task_{task.order}"
    prefix = "test_" if "test" in task.description.lower() else ""
    if prefix and not slug.startswith("test"):
        return f"{prefix}{slug}.py"
    return f"{slug}.py"


def _extract_code(raw: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    return fenced.group(1).strip() if fenced else raw.strip()


def _format_failure_analysis(analysis: dict) -> str:
    """Turns a Failure Analyzer result (see app.agents.failure_analyzer)
    into the repair-focused prompt block the Developer actually needs.

    Bug this fixes: previously, a retry only ever saw
    `previous_failure.error` — the raw QA output, truncated, with no root
    cause isolated, no affected function named, and no memory-informed
    hint — even though the Failure Analyzer node already computed all of
    that and then discarded it after logging it to the live event feed.
    The Developer was effectively re-diagnosing the same failure from
    scratch on every retry instead of being handed the diagnosis.
    """
    lines = [f"Root cause: {analysis.get('root_cause', 'unknown')}"]
    if analysis.get("affected_function") and analysis["affected_function"] != "unknown":
        lines.append(f"Affected function/section: {analysis['affected_function']}")
    if analysis.get("hint"):
        lines.append(f"Suggested fix: {analysis['hint']}")
    if analysis.get("repair_strategy"):
        lines.append(
            f"Repair strategy: {analysis['repair_strategy']} "
            f"(modify the specific issue above — do not rewrite unrelated code)"
        )

    similar = analysis.get("similar_fixes") or []
    if similar:
        examples = "\n".join(
            f"  - {s.get('problem_signature', '')[:80]} -> {s.get('solution_summary', '')[:120]}"
            for s in similar
        )
        lines.append(f"Similar failures fixed before:\n{examples}")

    return "\n".join(lines)


def generate_code(
    task: Task,
    previous_failure: TestResult | None = None,
    filename_hint: str | None = None,
    memory_context: list[dict] | None = None,
    original_request: str | None = None,
    failure_analysis: dict | None = None,
) -> tuple[str, str]:
    """Generates code for `task` and returns (code, filename). Does NOT write
    the file — that's the graph node's job, via the MCP write_file tool
    (app.mcp_client.call_tool), since this function is deliberately kept
    synchronous and side-effect-free for easy unit testing.

    If `previous_failure` is given, the prompt includes it so this is
    treated as a fix attempt rather than a first draft. If `failure_analysis`
    is ALSO given (the Failure Analyzer's output — see
    app.agents.failure_analyzer and app.pipeline.nodes.failure_analyzer_node),
    it takes precedence over the raw `previous_failure` text: a targeted
    root cause, affected function, and repair hint beat a truncated raw
    error blob for guiding a focused fix.

    If `memory_context` is given (a list of {problem_signature,
    solution_summary} dicts from app.memory.retrieve_similar), it's injected
    as few-shot context — past fixes for similar problems, from ANY project,
    not just this one — so the Developer doesn't start from a blank page on
    a bug the system has already solved before (Week 7 checkpoint).

    `original_request` (Phase 6, communication-mode ablation): the raw user
    request text, included ONLY when the caller passes it — which
    graph.py's developer_node does only when state.communication_mode ==
    "blackboard". This mirrors SE-Blackboard's (Liu et al., 2026) context
    functions: C^MP(Coder) = {out(Planner)} vs. C^BB(Coder) = {A, I}, where
    I is the original issue text. Passing None here reproduces AIOpsForge's
    original behavior — the Developer sees only task.description, i.e.
    Planner's already-summarized output, matching C^MP. Passing the raw
    request is the "blackboard" arm: the Developer can cross-reference
    detail the Planner's one-line task description may have compressed
    away (exact names, numbers, formats mentioned in the request but not
    repeated in the task list).
    """
    prompt = ""
    if original_request:
        prompt += (
            f"Original user request (for full context — the task below is "
            f"the Planner's breakdown of this; use this if it has detail "
            f"the task description doesn't):\n{original_request}\n\n"
        )

    if memory_context:
        examples = "\n\n".join(
            f"Similar past problem: {m['problem_signature']}\n"
            f"How it was fixed: {m['solution_summary']}"
            for m in memory_context
        )
        prompt += (
            f"Relevant past experience (from previously completed projects):\n"
            f"{examples}\n\n"
            f"Use this as a hint if it applies, but still solve THIS task correctly.\n\n"
        )

    prompt += f"Task: {task.description}"
    if previous_failure is not None:
        prompt += (
            f"\n\nYour previous attempt FAILED when tested. Fix it.\n"
            f"Test error:\n{previous_failure.error or previous_failure.output}"
        )
        if failure_analysis and failure_analysis.get("root_cause"):
            prompt += (
                f"\n\nFailure Analyzer diagnosis (use this to make a targeted "
                f"fix, not a rewrite):\n{_format_failure_analysis(failure_analysis)}"
            )

    raw = generate(prompt, system=DEVELOPER_SYSTEM_PROMPT)
    code = _extract_code(raw)
    filename = filename_hint or _suggest_filename(task)
    return code, filename
