from workspace import project_dir, safe_join


def write_file(project_id: str, path: str, content: str) -> str:
    """Write `content` to `path` (relative) inside the given project's
    workspace, creating any parent directories as needed. Overwrites if the
    file already exists. Returns the absolute path written.
    """
    pd = project_dir(project_id)
    target = safe_join(pd, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} bytes to {target}"
