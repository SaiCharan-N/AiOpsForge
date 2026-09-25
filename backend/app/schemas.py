"""Shared state schema for the agent graph (Day 4).

`AgentState` is the object that will flow through every LangGraph node from
Phase 2 onward — Planner reads `request` and writes `task_list`; Developer
reads `current_task` and writes `code`; QA reads `code` and writes
`test_result`, bumping `attempt_count` on failure. None of the agents exist
yet in Phase 1 — this is just the contract they'll all share, defined early
so nobody has to guess field names later.
"""
from typing import Optional

from pydantic import BaseModel, Field


class Task(BaseModel):
    order: int
    description: str
    status: str = "pending"  # pending | in_progress | passed | failed
    # Phase 7: "code" (default — single file/function, existing behavior)
    # or "webapp_scaffold" (multi-file FastAPI+Jinja2 project). Set by
    # app.agents.planner based on app.webapp_detect; read by graph.py to
    # route to app.agents.scaffolder instead of app.agents.developer, and
    # to the MCP check_webapp tool instead of run_tests.
    type: str = "code"


class TestResult(BaseModel):
    passed: bool
    output: str = ""
    error: Optional[str] = None


class AgentState(BaseModel):
    # Set once, at the start of a run.
    project_id: str
    request: str

    # Phase 6: which communication architecture this run uses, mirroring
    # SE-Blackboard's C^MP vs C^BB context functions (Liu et al., 2026).
    #   "message_passing" — the Developer sees ONLY the Planner's task
    #       description (out(Planner)); this was AIOpsForge's original,
    #       implicit behavior before this field existed.
    #   "blackboard" — the Developer ALSO receives the original raw
    #       request text, so it can cross-reference details the Planner's
    #       summarization may have dropped or paraphrased.
    # See app/agents/developer.py:generate_code for where this is applied.
    communication_mode: str = "blackboard"

    # Written by the Planner Agent (Phase 2, Week 2).
    task_list: list[Task] = Field(default_factory=list)

    # Orchestrator bookkeeping as it walks the task list.
    task_index: int = 0
    current_task: Optional[Task] = None

    # Written by the Developer Agent (Phase 2, Week 4). Filename the code
    # was written to, so QA and the next retry both know what to look at.
    code: Optional[str] = None
    code_filename: Optional[str] = None

    # Written by the QA Agent (Phase 2, Week 5).
    test_result: Optional[TestResult] = None

    # Incremented on every QA failure for the CURRENT task; reset to 0 when
    # moving to the next task. Capped at settings.max_qa_attempts (Day 24).
    attempt_count: int = 0

    # Set True once every task has been processed (passed or escalated).
    done: bool = False
    # Set True if ANY task exhausted its retries and needed a human.
    needs_human_review: bool = False

    # Set True when the Developer used a long-term memory hit for the
    # CURRENT task's first attempt (Week 7). Reset per task in next_task_node.
    memory_used: bool = False
    # Phase 6: IFS score (see app/ifs.py) for the most recently generated
    # code, re-set to None at the start of each new task by next_task_node.
    ifs_score: Optional[float] = None

    # Per-task outcomes, appended to as the graph runs — this is what the
    # final summary and the dashboard (Phase 3) will read.
    task_history: list[dict] = Field(default_factory=list)

    # Phase 11: Internal routing and state fields used by nodes but not
    # exposed in the final result or dashboard.
    _failure_analysis: Optional[dict] = None  # Analysis output from failure_analyzer_node
    _qa_decision: Optional[str] = None  # Route decision from qa_node (next_task, analyze_failure, escalate, retry)
    _last_escalated: bool = False  # Set True when a task is escalated for human review


class ProjectRequest(BaseModel):
    """Input to POST /request — a plain-English description of what to build."""
    request: str = Field(..., description="e.g. 'a function that reverses a string, with tests'")
    project_name: Optional[str] = None
    # Phase 6: "blackboard" (default) or "message_passing" — see AgentState.
    # Exposed here so the ablation experiments can drive it per-run without
    # touching code, e.g. {"request": "...", "communication_mode": "message_passing"}.
    communication_mode: str = "blackboard"


class ProjectResult(BaseModel):
    project_id: str
    done: bool
    needs_human_review: bool
    task_history: list[dict]


class GraphDemoRequest(BaseModel):
    """Input for the Day 3/5 'hello world' LangGraph demo — not part of the
    real agent pipeline, just proves LangGraph -> Ollama wiring works.
    """
    topic: str = Field(..., description="Any short topic to ask the local LLM about.")


class GraphDemoResponse(BaseModel):
    node_a_output: str
    node_b_output: str
