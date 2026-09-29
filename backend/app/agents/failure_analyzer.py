"""Failure Analyzer Agent (Phase 11).

Takes a QA failure and produces a targeted repair strategy:
  - Identifies the specific file/function that failed
  - Extracts the error message and root cause
  - Looks up similar failures in long-term memory
  - Recommends a focused repair (\"modify this file\", not \"regen everything\")

This agent runs BEFORE the retry developer step, so the Developer can do
targeted repairs instead of full regeneration.
"""
import logging
import re
from typing import Optional

from app.llm import generate_json, LLMOutputError
from app.memory import retrieve_similar
from app.schemas import TestResult

logger = logging.getLogger("aiopsforge.failure_analyzer")


FAILURE_ANALYZER_SYSTEM_PROMPT = """You are the Failure Analyzer in an automated software development pipeline.
Given a test failure or error message, your job is to:
1. Identify which file (e.g., "main.py", "test_app.py") failed
2. Extract the root cause (e.g., "AssertionError: expected 5 got 3")
3. Recommend the minimum change needed to fix it (e.g., "modify calculate() function")
4. Suggest a strategy: should we modify one file or regenerate the whole project?

Respond with ONLY a JSON object of this exact shape:
{
  "failed_file": "main.py",
  "error_type": "AssertionError",
  "root_cause": "expected 5 got 3",
  "affected_function": "calculate()",
  "repair_strategy": "modify",
  "priority": "high",
  "hint": "Check the off-by-one error in the loop"
}

Be concise and specific. If you cannot identify the file, use "unknown"."""


def analyze_failure(failure: TestResult, code_filename: str | None = None) -> dict:
    """
    Analyzes a QA failure and returns a repair strategy.
    
    Args:
        failure: TestResult with error message
        code_filename: The main code file that was tested (e.g., "app.py")
    
    Returns:
        dict with keys: failed_file, error_type, root_cause, affected_function,
                       repair_strategy, priority, hint, similar_fixes
    """
    if failure.passed:
        return {"analysis": "No failure to analyze"}
    
    # Extract file and error from failure message
    error_msg = failure.error or failure.output or "Unknown error"
    
    # Try to identify the failed file
    failed_file = _extract_failed_file(error_msg, code_filename)
    
    # Query memory for similar failures
    similar_fixes = []
    try:
        similar_fixes = retrieve_similar(
            error_msg[:100],
            top_k=2,
            return_context=True,
        )
        logger.info(f"found {len(similar_fixes)} similar fixes in memory")
    except Exception as e:
        logger.warning(f"memory lookup failed during analysis: {e}")
    
    # Build the analysis prompt
    analysis_prompt = f"""
Error message:
{error_msg}

Failed file (if known): {failed_file}
Main code file: {code_filename or 'unknown'}

Similar past fixes from memory:
{_format_similar_fixes(similar_fixes)}

Analyze this failure and provide a repair strategy.
"""
    
    # Call LLM for analysis
    try:
        analysis = generate_json(analysis_prompt, system=FAILURE_ANALYZER_SYSTEM_PROMPT)
    except LLMOutputError as e:
        logger.error(f"failure analyzer LLM failed: {e}")
        # Fallback analysis
        analysis = {
            "failed_file": failed_file,
            "error_type": _extract_error_type(error_msg),
            "root_cause": error_msg[:100],
            "affected_function": "unknown",
            "repair_strategy": "modify",
            "priority": "high",
            "hint": "Check the error message above",
        }
    
    # Attach similar fixes for the repair step to use
    analysis["similar_fixes"] = similar_fixes
    
    logger.info(
        f"failure analysis: file={analysis.get('failed_file')}, "
        f"strategy={analysis.get('repair_strategy')}, "
        f"similar_fixes={len(similar_fixes)}"
    )
    
    return analysis


def _extract_failed_file(error_msg: str, code_filename: str | None = None) -> str:
    """Tries to extract the filename from an error message.
    Format is often 'File "path/to/file.py", line 123, in function_name'
    """
    # Look for File "..." patterns
    match = re.search(r'File "([^"]+)"', error_msg)
    if match:
        path = match.group(1)
        # Return just the filename
        return path.split('/')[-1]
    
    # Fallback to provided code_filename
    if code_filename:
        return code_filename.split('/')[-1]
    
    return "unknown"


def _extract_error_type(error_msg: str) -> str:
    """Extracts the error type (e.g., 'AssertionError', 'ValueError')."""
    # Look for common Python error types
    match = re.search(r'(\w+Error|Exception):', error_msg)
    if match:
        return match.group(1)
    
    # Look for pytest FAILED keyword
    if "FAILED" in error_msg or "assert" in error_msg.lower():
        return "AssertionError"
    
    return "UnknownError"


def _format_similar_fixes(similar_fixes: list[dict]) -> str:
    """Formats similar fixes from memory for inclusion in the LLM prompt."""
    if not similar_fixes:
        return "(None)"
    
    lines = []
    for i, fix in enumerate(similar_fixes, 1):
        lines.append(
            f"{i}. Problem: {fix['problem_signature'][:80]}\n"
            f"   Solution: {fix['solution_summary'][:100]}\n"
            f"   Similarity: {fix['similarity']:.2f}"
        )
    
    return "\n".join(lines)
