"""QA Agent (Week 5 / Days 21-25).

Runs the Developer's generated code through pytest via the MCP run_tests
tool and turns the raw output into a structured pass/fail result the
orchestration graph can route on. Also generates a matching test file for
non-test tasks, since pytest needs something to actually run.
"""
import re

from app.schemas import TestResult

PASSED_RE = re.compile(r"PASSED:\s*(true|false)", re.IGNORECASE)
EXIT_CODE_RE = re.compile(r"EXIT_CODE:\s*(-?\d+)")


def parse_run_tests_output(raw: str) -> TestResult:
    """Parses the structured text the MCP run_tests tool returns
    (see mcp_server/server.py's run_tests) into a TestResult.
    """
    passed_match = PASSED_RE.search(raw)
    passed = bool(passed_match and passed_match.group(1).lower() == "true")

    output_marker = "OUTPUT:\n"
    output = raw.split(output_marker, 1)[1] if output_marker in raw else raw

    error = None
    if not passed:
        error = _extract_failure_summary(output)

    return TestResult(passed=passed, output=output.strip(), error=error)


def _extract_failure_summary(output: str) -> str:
    """Pulls out the most useful lines from pytest's verbose+short-tb output
    for feeding back to the Developer Agent, rather than the entire log.
    """
    lines = output.strip().splitlines()
    interesting = [
        line for line in lines
        if any(marker in line for marker in ("Error", "assert", "FAILED", "Traceback"))
    ]
    summary = "\n".join(interesting[-8:]) if interesting else output[-500:]
    return summary.strip() or "Tests failed with no captured output."
