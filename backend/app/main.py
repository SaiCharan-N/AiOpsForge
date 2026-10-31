"""FastAPI app. Phase 1 gave us /health + the LangGraph/Ollama wiring demo.
Phase 2 added the real pipeline: POST /request runs Planner -> Developer -> QA
end to end and persists the project, its tasks, and every agent run to
Postgres. Phase 3 adds an async version for the dashboard to poll, plus a
project download endpoint.
"""
import asyncio
import io
import logging
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from app.config import settings
from app.schemas import (
    EXPERIMENT_FACTORS, BatchRequest, ExperimentRequest, GraphDemoRequest, GraphDemoResponse, InterventionRequest, ProjectRequest,
    ProjectResult, UserFeedback,
)
from app import db, events, metrics_report, redis_client
from app.graph_demo import run_demo
from app import batch, experiments, scaffolds, webapp_detect
from app.pipeline import event_types as ev, intervention
from app.pipeline.graph import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aiopsforge")

app = FastAPI(title="AIOpsForge Backend", version="0.3.0-phase3")

# The dashboard (Week 9) runs on a different port during local dev — wide
# open CORS is fine for a student project's local dashboard, not something
# you'd want unmodified in front of a real public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _run_schema_migrations() -> None:
    """Best-effort schema fix-up (see db.run_schema_migrations's docstring
    for why this is needed even though db/init.sql already declares these
    columns). Logged and swallowed on failure — same philosophy as every
    other best-effort call in this app — so a slow-starting Postgres can't
    crash the whole backend; /health/deep will surface a real DB problem
    anyway.
    """
    try:
        db.run_schema_migrations()
        logger.info("schema migrations verified (memory_entries, tasks.type, failure_patterns, metrics_logs, interventions, batches, experiments)")
        # Batches/experiments still marked 'running' belonged to a process that no longer exists.
        stale = experiments.recover_stale()
        if any(stale.values()):
            logger.warning("marked interrupted by the restart: %s", stale)
    except Exception as exc:  # noqa: BLE001
        logger.warning("schema migration failed, continuing: %s", exc)


@app.get("/health")
def health():
    """Basic liveness check — always returns 200 if the process is up."""
    return {"status": "ok", "service": "aiopsforge-backend"}


@app.get("/health/deep")
def health_deep():
    """Checks every downstream dependency: Postgres, Redis, Ollama.
    Useful during Phase 1 to see exactly which piece isn't wired up yet.
    """
    checks = {}

    try:
        db.check_connection()
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001 — we want to report any failure, not crash
        checks["postgres"] = f"error: {exc}"

    try:
        redis_client.roundtrip_check()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc}"

    try:
        import requests
        from app.config import settings

        r = requests.get(f"{settings.ollama_base_url}/api/version", timeout=5)
        r.raise_for_status()
        checks["ollama"] = f"ok ({r.json().get('version', 'unknown version')})"
    except Exception as exc:  # noqa: BLE001
        checks["ollama"] = f"error: {exc}"

    overall = "ok" if all(v == "ok" or v.startswith("ok") for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}


@app.get("/db/tables")
def db_tables():
    """Lists tables created by db/init.sql — confirms Day 4's schema landed."""
    try:
        return {"tables": db.list_tables()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Postgres not reachable: {exc}") from exc


def _resolve_framework_choice(payload: ProjectRequest, query_framework: str | None) -> str | None:
    """The explicit framework for this request: ?framework= beats the body
    field; both go through the same validation. None = auto-detect."""
    try:
        return webapp_detect.validate_framework(query_framework) or payload.framework
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/frameworks")
def list_frameworks():
    """The web frameworks the pipeline can scaffold (use the `id` as
    `framework` on POST /request or /request/async, or omit it to auto-detect
    from the request text). `default` is used for a web app that names none."""
    return {"default": scaffolds.DEFAULT_FRAMEWORK,
            "frameworks": [f.public() for f in scaffolds.FRAMEWORKS.values()]}


@app.post("/request", response_model=ProjectResult)
async def submit_request(payload: ProjectRequest, framework: str | None = Query(None, description="fastapi | flask | django | express | next (omit to auto-detect)")):
    """The synchronous entry point (Week 2 onward): runs the full
    Planner -> Developer -> QA pipeline and blocks until it's done. Good for
    scripts and testing; the dashboard uses POST /request/async instead so
    it can show live progress while the pipeline runs.
    """
    if payload.interactive:
        # This endpoint blocks until the run ends and never returns the
        # project id, so nobody could answer a pause. Fail loudly instead of hanging.
        raise HTTPException(status_code=422, detail="interactive mode needs POST /request/async "
                            "(the project_id is returned immediately, so you can send feedback)")
    fw_choice = _resolve_framework_choice(payload, framework)
    project_id = str(uuid.uuid4())
    project_name = payload.project_name or payload.request[:60]

    try:
        db.create_project(project_id, project_name, payload.request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Could not create project record: {exc}") from exc

    try:
        result = await run_pipeline(project_id, payload.request, payload.communication_mode,
                                    framework=fw_choice)
    except Exception as exc:  # noqa: BLE001
        logger.exception("pipeline failed for project %s", project_id)
        db.update_project_status(project_id, "error")
        raise HTTPException(status_code=502, detail=f"Pipeline failed: {exc}") from exc

    try:
        db.update_project_status(
            project_id, "needs_human_review" if result.needs_human_review else "done"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to update final project status for %s: %s", project_id, exc)

    return ProjectResult(
        project_id=project_id,
        done=result.done,
        needs_human_review=result.needs_human_review,
        task_history=result.task_history,
        framework=result.framework,
    )


async def _run_pipeline_background(project_id: str, request: str, communication_mode: str = "blackboard",
                                   interactive: bool = False, pause_points: list[str] | None = None,
                                   framework: str | None = None) -> None:
    """Runs the pipeline and updates the project's final status — used as a
    FastAPI BackgroundTask by /request/async so the HTTP response returns
    immediately and the dashboard polls /projects/{id}/status for progress.
    """
    try:
        result = await run_pipeline(project_id, request, communication_mode, interactive, pause_points, framework)
        db.update_project_status(
            project_id,
            "cancelled" if result.aborted else "needs_human_review" if result.needs_human_review else "done",
        )
    except Exception:  # noqa: BLE001
        logger.exception("background pipeline run failed for project %s", project_id)
        try:
            db.update_project_status(project_id, "error")
        except Exception:  # noqa: BLE001
            pass


@app.post("/request/async")
async def submit_request_async(payload: ProjectRequest, background_tasks: BackgroundTasks,
                               framework: str | None = Query(None, description="fastapi | flask | django | express | next (omit to auto-detect)")):
    """Week 9: kicks off the pipeline in the background and returns
    immediately with a project_id — the dashboard then polls
    GET /projects/{project_id}/status to show live task-by-task progress.
    """
    fw_choice = _resolve_framework_choice(payload, framework)
    project_id = str(uuid.uuid4())
    project_name = payload.project_name or payload.request[:60]

    try:
        db.create_project(project_id, project_name, payload.request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Could not create project record: {exc}") from exc

    background_tasks.add_task(
        _run_pipeline_background, project_id, payload.request, payload.communication_mode,
        payload.interactive, payload.pause_points, fw_choice,
    )
    return {"project_id": project_id, "status": "building", "interactive": payload.interactive,
            "framework": fw_choice or webapp_detect.detect_framework(payload.request)}


@app.websocket("/ws/projects/{project_id}")
async def project_events_ws(websocket: WebSocket, project_id: str):
    """Phase 5: streams every Planner/Developer/QA event for one project as
    it happens — this is what the chat panel and the live flow diagram in
    the frontend both consume, instead of the old polling loop.

    A client that connects after the pipeline already started (or
    reconnects mid-run) still gets the full story: app.events.subscribe
    replays every past event for this project before switching to live.
    """
    await websocket.accept()
    queue = events.subscribe(project_id)
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        events.unsubscribe(project_id, queue)


@app.get("/projects/{project_id}/status")
def project_status(project_id: str):
    """Polled by the dashboard every couple of seconds. Returns the
    project's overall status plus every task's current status, in order —
    backed by app.db.upsert_task calls made incrementally inside the graph.
    """
    try:
        result = db.get_project_status(project_id)
    except Exception as exc:  # noqa: BLE001 — e.g. project_id isn't a valid UUID
        raise HTTPException(status_code=404, detail="No project with that id") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="No project with that id")
    # User intervention: is the run parked waiting for the user right now?
    result["intervention"] = intervention.status(project_id)
    result["paused"] = result["intervention"]["paused"]
    return result


# --------------------------------------------------------- intervention ----
# Multi-turn control of a RUNNING pipeline (app.pipeline.intervention).

def _project_exists(project_id: str) -> bool:
    try:
        return db.get_project_status(project_id) is not None
    except Exception:  # noqa: BLE001 — e.g. not a valid UUID
        return False


def _no_active_run(project_id: str) -> HTTPException:
    if not _project_exists(project_id):
        return HTTPException(status_code=404, detail="No project with that id")
    return HTTPException(status_code=409, detail="That project has no active run (it already finished, or never started)")


@app.post("/projects/{project_id}/intervention")
async def request_intervention(project_id: str, body: InterventionRequest):
    """Ask the running pipeline to pause the next time `pause_after`
    finishes ("next" = whichever node finishes first). The run then waits
    for POST /projects/{id}/feedback. `cancel=true` withdraws pending
    pause requests. Works for any run, interactive or not."""
    try:
        if body.cancel:
            ctl = intervention.cancel_pause_requests(project_id)
        else:
            ctl = intervention.request_pause(project_id, body.pause_after)
    except KeyError:
        raise _no_active_run(project_id) from None
    if not body.cancel:
        await events.emit(
            project_id, ev.PAUSE_REQUESTED, "system",
            "Pause requested — the run will stop for your input "
            + ("at the next step." if body.pause_after == "next" else f"after {body.pause_after.replace('_', ' ')} finishes."),
            {"pause_after": body.pause_after},
        )
        try:
            await asyncio.to_thread(db.log_intervention, project_id, "pause_request",
                                    body.pause_after, body.pause_after)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not log pause request for %s: %s", project_id, exc)
    return {"project_id": project_id, "pending_pause": sorted(ctl.pending), "paused": ctl.waiting is not None}


@app.post("/projects/{project_id}/feedback")
async def submit_feedback(project_id: str, body: UserFeedback):
    """Answer the pause the run is currently parked at: approve, revise
    (guidance / clarification / an edited task list at the plan review),
    skip the current task, or abort. 409 if the run isn't paused."""
    try:
        w = intervention.submit_feedback(project_id, intervention.Feedback(
            action=body.action, guidance=body.guidance, clarification=body.clarification, tasks=body.tasks))
    except KeyError:
        raise _no_active_run(project_id) from None
    except intervention.NotPaused:
        raise HTTPException(status_code=409, detail="The run is not waiting for feedback right now") from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"project_id": project_id, "accepted": True, "point": w.point, "action": body.action}


@app.get("/projects/{project_id}/interventions")
def project_interventions(project_id: str):
    """The project's intervention history: every pause request and every
    resolved pause (what was asked, the answer, how long the run waited)."""
    if not _project_exists(project_id):
        raise HTTPException(status_code=404, detail="No project with that id")
    try:
        return {"project_id": project_id, "interventions": db.list_interventions(project_id)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Could not read intervention history: {exc}") from exc


# --------------------------------------------------- batches & experiments ----
# Many runs at once, and controlled comparisons between them (app/batch.py, app/experiments.py).

def _csv_response(content: str, filename: str) -> Response:
    return Response(content=content, media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.post("/batch")
async def create_batch(body: BatchRequest):
    """Run several requests concurrently. Each item may be a plain string or an
    object with its own settings (framework, communication_mode, model,
    max_qa_attempts, use_memory); `defaults` fill in what an item leaves unset.
    Returns immediately with a batch_id; poll GET /batch/{id}, then read
    GET /batch/{id}/results (json or csv)."""
    try:
        items = batch.resolve_items(body.requests, body.defaults)
        batch_id = await batch.manager.start(
            items, name=body.name, max_concurrency=body.max_concurrency,
            config={"defaults": body.defaults.model_dump(exclude_none=True)})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — e.g. Postgres down
        raise HTTPException(status_code=503, detail=f"Could not create the batch: {exc}") from exc
    return {"batch_id": batch_id, "status": "running", "total": len(items),
            "status_url": f"/batch/{batch_id}", "results_url": f"/batch/{batch_id}/results"}


@app.get("/batch")
def list_batches(limit: int = Query(50, ge=1, le=200)):
    """Recent batches (newest first) with their progress."""
    return {"batches": _metrics_call(batch.list_batches, limit)}


@app.get("/batch/{batch_id}")
def batch_status(batch_id: str):
    """Live status: running / done / failed / cancelled / interrupted, progress,
    and the state of every project in the batch."""
    st = _metrics_call(batch.status, batch_id)
    if st is None:
        raise HTTPException(status_code=404, detail="No batch with that id")
    return st


@app.get("/batch/{batch_id}/results")
def batch_results(batch_id: str, format: str = Query("json", pattern="^(json|csv)$")):
    """The comparison table: one row per request (status, framework, tokens,
    latency, pass rate, cost, attempts, ...) plus aggregates. `format=csv`
    downloads the raw per-run rows."""
    res = _metrics_call(batch.results, batch_id)
    if res is None:
        raise HTTPException(status_code=404, detail="No batch with that id")
    if format == "csv":
        return _csv_response(batch.results_csv(res), f"batch-{batch_id[:8]}.csv")
    return res


@app.post("/batch/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    """Stops a running batch (runs in flight are cancelled; unstarted ones never start)."""
    if await batch.manager.cancel(batch_id):
        return {"batch_id": batch_id, "cancelled": True}
    if _metrics_call(batch.status, batch_id) is None:
        raise HTTPException(status_code=404, detail="No batch with that id")
    raise HTTPException(status_code=409, detail="That batch is not running")


@app.get("/experiments/presets")
def experiment_presets():
    """Ready-made experiment designs (including the three original scripts)."""
    return {"presets": [{"name": k, **{f: v for f, v in p.items()}} for k, p in experiments.PRESETS.items()],
            "factors": list(EXPERIMENT_FACTORS),
            "metrics": [{"key": k, "label": l, "kind": kd} for k, l, kd, _ in experiments.METRICS]}


@app.post("/experiments")
async def create_experiment(body: ExperimentRequest):
    """Start a controlled comparison — e.g. {"preset": "communication_mode"} or
    {"design": "ab", "factor": "max_qa_attempts", "levels": [1, 3, 5], "requests": [...], "repetitions": 3}.
    Returns the experiment_id immediately; poll GET /experiments/{id}; when it
    is done GET /experiments/{id}/results has the statistics."""
    try:
        exp_id = await experiments.manager.start(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Could not start the experiment: {exc}") from exc
    return {"experiment_id": exp_id, "status": "running", "status_url": f"/experiments/{exp_id}",
            "results_url": f"/experiments/{exp_id}/results"}


@app.get("/experiments")
def list_experiments(limit: int = Query(50, ge=1, le=200)):
    return {"experiments": _metrics_call(experiments.list_experiments, limit)}


@app.get("/experiments/{experiment_id}")
def experiment_status(experiment_id: str):
    st = _metrics_call(experiments.status, experiment_id)
    if st is None:
        raise HTTPException(status_code=404, detail="No experiment with that id")
    return st


@app.get("/experiments/{experiment_id}/results")
def experiment_results(experiment_id: str, format: str = Query("json", pattern="^(json|csv|latex)$")):
    """The stored result: per-condition summaries, statistical comparisons
    (Welch / Mann-Whitney / Fisher with Holm correction), a plain-language
    verdict and warnings. `format=csv` = raw per-run data for re-analysis,
    `format=latex` = booktabs tables for a paper. `result` is null while running."""
    if format == "csv":
        content = _metrics_call(experiments.runs_csv, experiment_id)
        if content is None:
            raise HTTPException(status_code=404, detail="No experiment with that id")
        return _csv_response(content, f"experiment-{experiment_id[:8]}.csv")
    if format == "latex":
        content = _metrics_call(experiments.latex, experiment_id)
        if content is None:
            raise HTTPException(status_code=404, detail="No experiment with that id")
        return Response(content=content, media_type="application/x-tex",
                        headers={"Content-Disposition": f"attachment; filename=experiment-{experiment_id[:8]}.tex"})
    res = _metrics_call(experiments.results, experiment_id)
    if res is None:
        raise HTTPException(status_code=404, detail="No experiment with that id")
    return res


@app.post("/experiments/{experiment_id}/cancel")
async def cancel_experiment(experiment_id: str):
    if await experiments.manager.cancel(experiment_id):
        return {"experiment_id": experiment_id, "cancelled": True}
    if _metrics_call(experiments.status, experiment_id) is None:
        raise HTTPException(status_code=404, detail="No experiment with that id")
    raise HTTPException(status_code=409, detail="That experiment is not running")


# ------------------------------------------------------------- metrics ----
# Cost & performance metrics (app.pipeline.metrics writes them,
# app.metrics_report reads them). All read-only, all JSON except /metrics/export.

def _metrics_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — e.g. Postgres down, or an id that isn't a valid UUID
        logger.warning("metrics query failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Metrics unavailable: {exc}") from exc


@app.get("/projects/{project_id}/metrics")
def project_metrics(project_id: str):
    """Tokens & estimated cost per agent, latency breakdown (LLM / tools /
    DB / embeddings / transport overhead), the per-node timeline used for
    the flamegraph, tool usage and per-task outcomes for ONE project."""
    result = _metrics_call(metrics_report.project_metrics, project_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No metrics recorded for that project id")
    return result


@app.get("/projects/{project_id}/ifs")
def project_ifs(project_id: str):
    """Information Fidelity Score records of ONE project: the Planner-stage score, and per task
    the Developer's score on its last attempt, each with the substring and the strict matcher and
    the list of request entities that were lost. Every attempt's record is in `records`."""
    result = _metrics_call(metrics_report.ifs_records, project_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No IFS records for that project id "
                            "(the request may contain no extractable technical entities)")
    return result


@app.get("/metrics/projects")
def metrics_projects(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                     status: str | None = None, communication_mode: str | None = None):
    """One summary row per project (tokens, cost, latency, pass rate, attempts), newest first."""
    return {"projects": _metrics_call(metrics_report.list_projects, limit, offset, status, communication_mode)}


@app.get("/metrics/compare")
def metrics_compare(ids: str = Query(..., description="comma-separated project ids; the first is the baseline")):
    """Side-by-side metrics for 2+ projects with % change vs the first."""
    id_list = [i.strip() for i in ids.split(",") if i.strip()][:6]
    if len(id_list) < 2:
        raise HTTPException(status_code=422, detail="Provide at least two project ids")
    return _metrics_call(metrics_report.compare, id_list)


@app.get("/metrics/summary")
def metrics_summary(limit: int = Query(500, ge=1, le=5000)):
    """Pass rate, attempts, timeout and escalation rate by task type and by
    model; repeat-error rate and the effect of similar-fix memory hits."""
    return _metrics_call(metrics_report.success_summary, limit)


@app.get("/metrics/models")
def metrics_models():
    """Per-model speed (ms per output token, generation tokens/s, cold loads)
    and task success — which model performs best, and which is slower."""
    return _metrics_call(metrics_report.model_stats)


@app.get("/metrics/regression")
def metrics_regression(model: str | None = None, window: int = Query(5, ge=1, le=50),
                       threshold: float = Query(0.2, ge=0.0, le=5.0)):
    """Performance-regression check: recent vs earlier ms-per-output-token for a model."""
    return _metrics_call(metrics_report.regression, model, window, threshold)


@app.get("/metrics/export")
def metrics_export(format: str = Query("csv", pattern="^(csv|json)$"), limit: int = Query(500, ge=1, le=5000)):
    """Downloads the project summary table (all fields) as CSV or JSON."""
    rows = _metrics_call(metrics_report.list_projects, limit, 0, None, None)
    if format == "json":
        return {"projects": rows}
    return Response(
        content=metrics_report.to_csv(rows), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=aiopsforge-metrics.csv"},
    )


@app.get("/projects/{project_id}/download")
def download_project(project_id: str):
    """Zips the project's workspace directory and returns it. Requires the
    backend container to have the shared `workspace` volume mounted
    (read-only) — see docker-compose.yml.
    """
    project_dir = Path(settings.workspace_root) / project_id
    if not project_dir.is_dir():
        raise HTTPException(status_code=404, detail="No workspace found for that project id")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in project_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(project_dir))
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={project_id}.zip"},
    )


@app.post("/demo/graph-run", response_model=GraphDemoResponse)
def demo_graph_run(payload: GraphDemoRequest):
    """Day 5 checkpoint: runs the 2-node LangGraph demo, node_b calls Ollama."""
    try:
        result = run_demo(payload.topic)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Graph run failed: {exc}") from exc
    return GraphDemoResponse(
        node_a_output=result["node_a_output"],
        node_b_output=result["node_b_output"],
    )
