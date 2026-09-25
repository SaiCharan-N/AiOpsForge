import subprocess

from workspace import project_dir


def run_shell_command(project_id: str, command: str, timeout: int = 30) -> str:
    """Run a shell command inside the given project's workspace directory.
    Returns a formatted block with the exit code, stdout, and stderr.
    """
    pd = project_dir(project_id)
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(pd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"exit_code: -1 (timed out after {timeout}s)\nstdout:\n\nstderr:\nCommand timed out"

    return (
        f"exit_code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
