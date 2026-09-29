"""Shared per-project workspace path handling — the filesystem-level
isolation mechanism every tool in tools/ builds on. Split out from
server.py so tool modules can import it without importing the FastMCP app
itself (avoids a circular import: server.py imports tools, tools need this).
"""
import os
from pathlib import Path

WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", "/workspace")).resolve()
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


def project_dir(project_id: str) -> Path:
    """Resolves a project's workspace dir and guarantees it can't escape
    WORKSPACE_ROOT (blocks '../' style path traversal in project_id itself).
    """
    candidate = (WORKSPACE_ROOT / project_id).resolve()
    if WORKSPACE_ROOT not in candidate.parents and candidate != WORKSPACE_ROOT:
        raise ValueError(f"Invalid project_id: {project_id!r}")
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def safe_join(base: Path, relative_path: str) -> Path:
    """Joins relative_path onto base and guarantees the result stays inside
    base (blocks '../../etc/passwd' style traversal in the file path).
    """
    target = (base / relative_path).resolve()
    if base not in target.parents and target != base:
        raise ValueError(f"Path escapes project workspace: {relative_path!r}")
    return target
