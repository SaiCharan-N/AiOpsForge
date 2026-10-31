"""Shared state schema for the agent graph (Day 4).

`AgentState` is the object that will flow through every LangGraph node from
Phase 2 onward — Planner reads `request` and writes `task_list`; Developer
reads `current_task` and writes `code`; QA reads `code` and writes
`test_result`, bumping `attempt_count` on failure. None of the agents exist
yet in Phase 1 — this is just the contract they'll all share, defined early
so nobody has to guess field names later.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


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
    # Which framework this task targets ("fastapi" | "flask" | "django" |
    # "express" | "next"), or None if the request didn't name one and the
    # task isn't a web app. Set by the Planner (explicit choice > detected
    # from the request); read by the scaffolder, the Developer's prompt and QA.
    framework: Optional[str] = None


class StackFrame(BaseModel):
    """One frame of a parsed traceback (pytest short format or a plain
    Python traceback)."""
    file: str
    line: Optional[int] = None
    function: Optional[str] = None
    source: Optional[str] = None
    project_frame: bool = True  # False for site-packages / stdlib / pytest internals
    variables: dict[str, str] = Field(default_factory=dict)  # locals at this frame (pytest --showlocals)


class ExecutionContext(BaseModel):
    """The raw circumstances of one QA run — what ran, how, and what it
    printed. Kept alongside FailureContext (the parsed interpretation) so
    memory entries and the runs table can record both."""
    task_type: str = "code"
    qa_tool: str = "run_tests"
    deterministic: bool = False   # True if the failure came from the pre-flight check
    filename: Optional[str] = None
    attempt: int = 1
    model: str = ""
    exit_code: Optional[int] = None
    duration_ms: Optional[int] = None
    output_tail: str = ""         # last ~2000 chars of the QA output
    traceback: str = ""           # the failing block's traceback text
    variables: dict[str, str] = Field(default_factory=dict)  # locals at the failing frame


class FailureContext(BaseModel):
    """Structured interpretation of one QA failure (see
    app.agents.failure_parser). This is what the Failure Analyzer, the
    Developer's retry prompt, the live UI and long-term memory all read —
    instead of each re-parsing a truncated raw error string."""
    error_type: str = "UnknownError"
    # syntax | import | name | type | attribute | lookup | value | logic |
    # arithmetic | recursion | timeout | missing_file | config | runtime | unknown
    error_category: str = "unknown"
    message: str = ""
    failed_file: Optional[str] = None
    line: Optional[int] = None
    function: Optional[str] = None            # function enclosing the failing line
    affected_function: Optional[str] = None   # function most likely to need the fix
    code_location: str = ""                   # e.g. "calc.py:5 in avg"
    frames: list[StackFrame] = Field(default_factory=list)
    failing_tests: list[str] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    snippet: str = ""                         # line-numbered source around the failing line
    suspect_source: str = ""                  # line-numbered source of affected_function
    suspect_file: Optional[str] = None
    phase: str = "test"                       # test | collection | preflight | boot | timeout
    signature: str = ""                       # normalized, for repeat-error detection + memory lookup
    hint: str = ""                            # specific, deterministic repair hint
    hint_confidence: str = "low"              # low | medium | high
    raw_excerpt: str = ""


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
    # Phase 18: strict-matcher score of the same output, and the Planner-stage score
    # (see app.ifs.record_ifs).
    ifs_strict: Optional[float] = None
    ifs_planner: Optional[float] = None

    # Per-task outcomes, appended to as the graph runs — this is what the
    # final summary and the dashboard (Phase 3) will read.
    task_history: list[dict] = Field(default_factory=list)

    # User intervention (app.pipeline.intervention): everything the user
    # told the pipeline mid-run — [{point, action, guidance, clarification,
    # scope: "project"|"task", task_order, at}] — and which pause point the
    # run is currently parked at (None when running).
    user_feedback: list[dict] = Field(default_factory=list)
    paused_at: Optional[str] = None
    # True if the user chose "abort" at a pause point; the partial results
    # (task_history, files already written) are kept.
    aborted: bool = False

    # Framework the run targets: the user's explicit choice, else detected
    # from the request, else (for a web app with none named) "fastapi". None
    # for plain code requests. Set by the Planner node.
    framework: Optional[str] = None
    # The user's EXPLICIT framework choice (API param / UI select), or None to
    # auto-detect. Input to the Planner; `framework` above is the outcome.
    framework_choice: Optional[str] = None


class ProjectRequest(BaseModel):
    """Input to POST /request — a plain-English description of what to build."""
    request: str = Field(..., description="e.g. 'a function that reverses a string, with tests'")
    project_name: Optional[str] = None
    # Phase 6: "blackboard" (default) or "message_passing" — see AgentState.
    # Exposed here so the ablation experiments can drive it per-run without
    # touching code, e.g. {"request": "...", "communication_mode": "message_passing"}.
    communication_mode: str = "blackboard"
    # Multi-turn mode: pause for the user's review at the chosen points
    # (see app.pipeline.intervention). Off by default so every existing
    # caller — scripts, experiments, the sync endpoint — behaves as before.
    interactive: bool = False
    # Subset of ["planner", "developer", "qa_failure"]; None = all three.
    pause_points: Optional[list[str]] = None
    # Explicit framework (fastapi | flask | django | express | next). None /
    # "auto" = detect it from the request text. Choosing one makes the
    # request a web app in that framework. Also accepted as ?framework= on
    # POST /request and /request/async (the query parameter wins).
    framework: Optional[str] = None

    @field_validator("framework")
    @classmethod
    def _valid_framework(cls, v):
        from app.webapp_detect import validate_framework  # local: webapp_detect imports app.scaffolds
        return validate_framework(v)

    @field_validator("pause_points")
    @classmethod
    def _valid_points(cls, v):
        if v is None:
            return v
        bad = [p for p in v if p not in ("planner", "developer", "qa_failure")]
        if bad:
            raise ValueError(f"unknown pause point(s) {bad}; valid: planner, developer, qa_failure")
        return v


COMMUNICATION_MODES = ("blackboard", "message_passing")
EXPERIMENT_FACTORS = ("communication_mode", "framework", "model", "max_qa_attempts", "use_memory",
                      "failure_recovery", "temperature", "num_ctx")
EXPERIMENT_DESIGNS = ("ab", "repeat", "sequence", "mcp_overhead")


class RunSettings(BaseModel):
    """Everything that can differ from one run to the next. Unset = inherit
    (batch defaults, then the system default)."""
    communication_mode: Optional[Literal["blackboard", "message_passing"]] = None
    framework: Optional[str] = None            # fastapi | flask | django | express | next (None = auto-detect)
    model: Optional[str] = Field(None, max_length=100, description="Ollama model for this run")
    max_qa_attempts: Optional[int] = Field(None, ge=1, le=10, description="QA attempts per task before escalating")
    use_memory: Optional[bool] = Field(None, description="False = long-term memory is neither read nor written")
    failure_recovery: Optional[bool] = Field(None, description="False = retries see only the raw QA error (no diagnosis / previous code / hints)")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Ollama sampling temperature")
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1, description="Ollama sampling seed")
    num_ctx: Optional[int] = Field(None, ge=512, le=131072, description="Ollama context window (tokens)")

    @field_validator("framework")
    @classmethod
    def _fw(cls, v):
        from app.webapp_detect import validate_framework
        return validate_framework(v)

    @field_validator("model")
    @classmethod
    def _model(cls, v):
        if v is None:
            return v
        from app import run_config
        return run_config.validate({"model": v})["model"]


class BatchItem(RunSettings):
    request: str = Field(..., min_length=1, max_length=4000)
    project_name: Optional[str] = Field(None, max_length=120)


class BatchRequest(BaseModel):
    """POST /batch — run several requests concurrently. Items may be plain
    strings or objects with their own settings; `defaults` fill what an item
    leaves unset."""
    name: Optional[str] = Field(None, max_length=120)
    requests: list[BatchItem | str] = Field(..., min_length=1)
    defaults: RunSettings = Field(default_factory=RunSettings)
    max_concurrency: Optional[int] = Field(None, ge=1, le=20, description="default: BATCH_MAX_CONCURRENCY")


class ExperimentRequest(BaseModel):
    """POST /experiments — see app.experiments for the four designs.

    ab            the same requests under each level of one `factor`
                  (e.g. communication_mode: blackboard vs message_passing)
    repeat        every request run twice back to back (memory speed-up)
    sequence      requests run one after another (does a similar-task sequence get easier?)
    mcp_overhead  MCP tool-call latency vs a direct file write

    `preset` fills in a ready-made design (and default requests); anything you
    set explicitly wins over the preset."""
    preset: Optional[str] = None
    name: Optional[str] = Field(None, max_length=120)
    design: Optional[Literal["ab", "repeat", "sequence", "mcp_overhead"]] = None
    requests: Optional[list[str]] = Field(None, max_length=50)
    factor: Optional[Literal["communication_mode", "framework", "model", "max_qa_attempts", "use_memory",
                             "failure_recovery", "temperature", "num_ctx"]] = None
    levels: Optional[list[Any]] = Field(None, description="values of `factor`; the FIRST is the baseline")
    repetitions: int = Field(1, ge=1, le=10)
    base: RunSettings = Field(default_factory=RunSettings, description="settings shared by every arm")
    max_concurrency: Optional[int] = Field(None, ge=1, le=20)
    alpha: float = Field(0.05, gt=0, lt=0.5)
    calls: Optional[int] = Field(None, ge=2, le=200, description="mcp_overhead: number of tool calls")


class InterventionRequest(BaseModel):
    """POST /projects/{id}/intervention — ask the running pipeline to pause
    the next time the named node finishes ("next" = whichever finishes
    first). `cancel=true` withdraws pending pause requests instead."""
    pause_after: Literal["next", "planner", "developer", "qa", "failure_analyzer", "next_task"] = "next"
    cancel: bool = False


class UserFeedback(BaseModel):
    """POST /projects/{id}/feedback — the user's answer at a pause point.

    action: approve   continue as-is
            revise    apply `guidance` / `clarification` (or, at the plan,
                      the edited `tasks`) — re-plans / regenerates at the
                      planner and first-file pause points, and is added to
                      the Developer's context everywhere else
            skip_task give up on the current task (flags it for review)
            abort     stop the whole run
    """
    action: Literal["approve", "revise", "skip_task", "abort"] = "approve"
    guidance: Optional[str] = Field(None, max_length=4000, description="'Actually, split it this way…'")
    clarification: Optional[str] = Field(None, max_length=4000, description="'I meant X, not Y'")
    tasks: Optional[list[str]] = Field(None, max_length=12, description="edited plan (planner pause only)")

    @field_validator("tasks")
    @classmethod
    def _tasks_ok(cls, v):
        if v is None:
            return v
        cleaned = [t.strip() for t in v if t and t.strip()]
        if not cleaned:
            raise ValueError("tasks must contain at least one non-empty task")
        if any(len(t) > 1000 for t in cleaned):
            raise ValueError("each task must be at most 1000 characters")
        return cleaned


class ProjectResult(BaseModel):
    project_id: str
    done: bool
    needs_human_review: bool
    task_history: list[dict]
    framework: Optional[str] = None


class GraphDemoRequest(BaseModel):
    """Input for the Day 3/5 'hello world' LangGraph demo — not part of the
    real agent pipeline, just proves LangGraph -> Ollama wiring works.
    """
    topic: str = Field(..., description="Any short topic to ask the local LLM about.")


class GraphDemoResponse(BaseModel):
    node_a_output: str
    node_b_output: str
