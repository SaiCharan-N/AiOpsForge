#!/usr/bin/env python3
"""Phase 6 checkpoint: does the IFS metric actually behave sensibly?

No app.* imports beyond ifs.py itself needed — this module has zero
dependencies (stdlib re only), so no mcp/langfuse stubbing required here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app import ifs  # noqa: E402

REQUEST = (
    'Build an ISBN-13 validator called `validate_isbn13`. It must reject '
    'any input that mixes hyphens and spaces, and the weight pattern is '
    '1,3,1,3,... not the ISBN-10 pattern. Save it as isbn_validator.py.'
)


def check_extracts_expected_entities():
    entities = ifs.extract_entities(REQUEST)
    # Not asserting the exact set (rule-based extraction is intentionally
    # loose) — asserting the entities a paper reviewer would expect to see.
    expected_subset = {"validate_isbn13", "isbn_validator.py"}
    lower_entities = {e.lower() for e in entities}
    missing = {e for e in expected_subset if e.lower() not in lower_entities}
    assert not missing, f"expected to extract {expected_subset}, missing {missing} (got {entities})"
    assert any("ISBN-13" in e or "ISBN-10" in e for e in entities), (
        f"expected a hyphenated identifier like ISBN-13/ISBN-10, got {entities}"
    )
    print(f"[ok] extracted {len(entities)} entities from the request: {sorted(entities)}")


def check_full_preservation_scores_1():
    # Agent output that contains every entity from the request verbatim.
    perfect_output = (
        "def validate_isbn13(code):\n"
        "    '''Validates ISBN-13, rejecting mixed hyphens/spaces. "
        "Not the ISBN-10 pattern.'''\n"
        "# saved as isbn_validator.py\n"
    )
    score = ifs.compute_ifs(REQUEST, perfect_output)
    assert score is not None and score == 1.0, f"expected 1.0, got {score}"
    print(f"[ok] output containing every entity scores {score}")


def check_partial_preservation_scores_between_0_and_1():
    # Output that only kept the function name, dropped everything else
    # (filename, ISBN-13/10 distinction) — simulates a lossy summarization.
    lossy_output = "def validate_isbn13(code):\n    pass\n"
    score = ifs.compute_ifs(REQUEST, lossy_output)
    assert score is not None and 0.0 < score < 1.0, f"expected a partial score, got {score}"
    print(f"[ok] partially-preserving output scores {score:.2f} (between 0 and 1)")


def check_no_preservation_scores_0():
    unrelated_output = "def add(a, b):\n    return a + b\n"
    score = ifs.compute_ifs(REQUEST, unrelated_output)
    assert score == 0.0, f"expected 0.0, got {score}"
    print(f"[ok] unrelated output scores {score}")


def check_entity_free_request_returns_none_not_zero():
    vague_request = "write something that does the thing we talked about"
    score = ifs.compute_ifs(vague_request, "def f(): pass")
    assert score is None, (
        "a request with no extractable entities should return None (undefined), "
        "not 0.0 — 0.0 would misleadingly read as total information loss"
    )
    print("[ok] entity-free request correctly returns None, not a misleading 0.0")


def check_blackboard_beats_message_passing_on_a_realistic_example():
    """Not a claim about the real system's behavior — just confirms the
    metric responds in the expected DIRECTION: an output built with access
    to the full original request should score >= one built from a lossy
    one-line task summary, on a case where the task summary drops detail.
    """
    task_summary_only = "def validate_isbn13(code):\n    # basic isbn check\n    pass\n"
    with_original_request = (
        "def validate_isbn13(code):\n"
        "    '''ISBN-13 validator. Rejects mixed hyphens/spaces. "
        "Uses 1,3,1,3 weights, not ISBN-10 weights.'''\n"
    )
    mp_score = ifs.compute_ifs(REQUEST, task_summary_only)
    bb_score = ifs.compute_ifs(REQUEST, with_original_request)
    assert bb_score > mp_score, (
        f"expected blackboard-style output to score higher, got mp={mp_score}, bb={bb_score}"
    )
    print(f"[ok] richer-context output scores higher ({bb_score:.2f} vs {mp_score:.2f}), as expected")


if __name__ == "__main__":
    check_extracts_expected_entities()
    check_full_preservation_scores_1()
    check_partial_preservation_scores_between_0_and_1()
    check_no_preservation_scores_0()
    check_entity_free_request_returns_none_not_zero()
    check_blackboard_beats_message_passing_on_a_realistic_example()
    print("\nAll IFS metric checks passed.")
