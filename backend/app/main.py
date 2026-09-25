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

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import text

from app.config import settings
from app.schemas import (
    GraphDemoRequest, GraphDemoResponse, ProjectRequest, ProjectResult,
)
from app import db, events, redis_client
from app.graph_demo import run_demo
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
    """Best-effort schema fix-up (see db.ensure_memory_schema's docstring
    for why this is needed even though db/init.sql already declares these
    columns). Logged and swallowed on failure — same philosophy as every
    other best-effort call in this app — so a slow-starting Postgres can't
    crash the whole backend; /health/deep will surface a real DB problem
    anyway.
    """
    try:
        db.ensure_memory_schema()
        logger.info("memory_entries schema verified (embedding, execution_context present)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("memory_entries schema migration failed, continuing: %s", exc)


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


@app.post("/request", response_model=ProjectResult)
async def submit_request(payload: ProjectRequest):
    """The synchronous entry point (Week 2 onward): runs the full
    Planner -> Developer -> QA pipeline and blocks until it's done. Good for
    scripts and testing; the dashboard uses POST /request/async instead so
    it can show live progress while the pipeline runs.
    """
    project_id = str(uuid.uuid4())
    project_name = payload.project_name or payload.request[:60]

    try:
        db.create_project(project_id, project_name, payload.request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Could not create project record: {exc}") from exc

    try:
        result = await run_pipeline(project_id, payload.request, payload.communication_mode)
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
    )


async def _run_pipeline_background(project_id: str, request: str, communication_mode: str = "blackboard") -> None:
    """Runs the pipeline and updates the project's final status — used as a
    FastAPI BackgroundTask by /request/async so the HTTP response returns
    immediately and the dashboard polls /projects/{id}/status for progress.
    """
    try:
        result = await run_pipeline(project_id, request, communication_mode)
        db.update_project_status(
            project_id, "needs_human_review" if result.needs_human_review else "done"
        )
    except Exception:  # noqa: BLE001
        logger.exception("background pipeline run failed for project %s", project_id)
        try:
            db.update_project_status(project_id, "error")
        except Exception:  # noqa: BLE001
            pass


@app.post("/request/async")
async def submit_request_async(payload: ProjectRequest, background_tasks: BackgroundTasks):
    """Week 9: kicks off the pipeline in the background and returns
    immediately with a project_id — the dashboard then polls
    GET /projects/{project_id}/status to show live task-by-task progress.
    """
    project_id = str(uuid.uuid4())
    project_name = payload.project_name or payload.request[:60]

    try:
        db.create_project(project_id, project_name, payload.request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Could not create project record: {exc}") from exc

    background_tasks.add_task(_run_pipeline_background, project_id, payload.request, payload.communication_mode)
    return {"project_id": project_id, "status": "building"}


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
    return result


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
