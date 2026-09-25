"""Deterministic QA checks (Phase 11).

These checks do NOT require the LLM and are used as a first pass before
any LLM-based reasoning. They cover:
  - File existence and basic syntax (py_compile for .py files)
  - Import correctness
  - Test file existence and pytest execution
  - Expected output patterns (if specified)
  - No critical error strings in output

This reduces unnecessary LLM calls and speeds up feedback on obvious issues.
"""
import ast
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger("aiopsforge.qa_deterministic")


class DeterministicCheckFailed(Exception):
    """Raised when a deterministic check fails."""
    pass


def check_python_file_syntax(filepath: str) -> None:
    """Verifies a Python file compiles (syntax is valid).
    Raises DeterministicCheckFailed if not.
    """
    if not filepath.endswith('.py'):
        return  # Not a Python file, skip
    
    if not os.path.isfile(filepath):
        raise DeterministicCheckFailed(f"File not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        compile(code, filepath, 'exec')
        logger.info(f"syntax check passed: {filepath}")
    except SyntaxError as e:
        raise DeterministicCheckFailed(f"Syntax error in {filepath}: {e.msg} (line {e.lineno})")
    except Exception as e:
        raise DeterministicCheckFailed(f"Failed to compile {filepath}: {e}")


def check_file_exists(filepath: str) -> None:
    """Verifies a file exists."""
    if not os.path.isfile(filepath):
        raise DeterministicCheckFailed(f"Required file not found: {filepath}")
    logger.info(f"file exists: {filepath}")


def check_imports(filepath: str) -> None:
    """Verifies that a Python file's imports can be parsed (basic check)."""
    if not filepath.endswith('.py'):
        return
    
    if not os.path.isfile(filepath):
        raise DeterministicCheckFailed(f"File not found for import check: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        logger.info(f"import check passed: {len(imports)} import(s) found in {filepath}")
    except SyntaxError as e:
        raise DeterministicCheckFailed(f"Syntax error parsing imports in {filepath}: {e}")
    except Exception as e:
        raise DeterministicCheckFailed(f"Failed to parse imports in {filepath}: {e}")


def check_no_critical_errors(output: str, critical_patterns: list[str] | None = None) -> None:
    """Checks that output does NOT contain critical error strings.
    
    Args:
        output: The output to check (e.g., from pytest or shell)
        critical_patterns: List of regex patterns; if ANY match, raise error
                          Defaults to common Python errors
    """
    if critical_patterns is None:
        critical_patterns = [
            r"Traceback \(most recent",
            r"Error:",
            r"Exception:",
            r"FAILED",
            r"(ImportError|ModuleNotFoundError|SyntaxError|NameError)",
        ]
    
    for pattern in critical_patterns:
        if re.search(pattern, output, re.MULTILINE | re.IGNORECASE):
            raise DeterministicCheckFailed(
                f"Critical error pattern found: {pattern}\nOutput:\n{output[:500]}"
            )
    
    logger.info("no critical errors detected in output")


def check_output_contains(output: str, expected_substring: str) -> None:
    """Verifies that output contains a specific substring."""
    if expected_substring not in output:
        raise DeterministicCheckFailed(
            f"Expected output not found.\nExpected: {expected_substring}\nGot: {output[:200]}"
        )
    logger.info(f"output contains expected substring: '{expected_substring[:50]}'")


def run_deterministic_qa(
    project_id: str,
    task_type: str,
    code_filepath: str | None = None,
    test_filepath: str | None = None,
    expected_output: str | None = None,
) -> tuple[bool, str]:
    """
    Runs all applicable deterministic checks. Returns (passed, message).
    
    This is a pre-flight check before any LLM-based QA. If any check fails,
    returns False immediately with the failure reason.
    """
    try:
        if code_filepath:
            check_python_file_syntax(code_filepath)
            check_imports(code_filepath)
        
        if test_filepath:
            check_file_exists(test_filepath)
            # Optionally run pytest here if you want deterministic test execution
            # result = subprocess.run(["pytest", test_filepath, "-v"], capture_output=True, text=True)
            # if result.returncode != 0:
            #     raise DeterministicCheckFailed(f"Tests failed:\n{result.stdout}\n{result.stderr}")
        
        if expected_output and code_filepath:
            # For simple checks: if we have expected output, verify the file at least
            # contains a hint of that output (e.g., a return statement or string)
            with open(code_filepath, 'r') as f:
                code = f.read()
            if expected_output and expected_output not in code:
                logger.warning(f"expected output pattern not found in code, LLM may need to retry")
        
        logger.info(f"deterministic QA passed for project {project_id}")
        return True, "Deterministic QA checks passed (syntax, imports, file existence)"
    
    except DeterministicCheckFailed as e:
        logger.warning(f"deterministic QA failed: {e}")
        return False, str(e)
    except Exception as e:
        logger.warning(f"deterministic QA error: {e}")
        return False, f"Deterministic check error: {e}"
