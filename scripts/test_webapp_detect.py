#!/usr/bin/env python3
"""Phase 7 checkpoint: does the webapp-request detector fire on the phrasing
we actually expect people to use, and stay quiet on plain script requests?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.webapp_detect import is_webapp_request  # noqa: E402

SHOULD_MATCH = [
    "Build a website for a bakery with a homepage and a menu page",
    "Make a simple web app for tracking daily habits",
    "I need a landing page for my startup",
    "create a flask app that shows a list of books",
    "build a fastapi app with a login page",
    "Can you build a to-do list website?",
]

SHOULD_NOT_MATCH = [
    "a function that checks if a string is a palindrome, with a test",
    "write a script that renames all .txt files in a folder",
    "implement a binary search algorithm in Python",
    "a class that represents a bank account with deposit and withdraw",
]


def check_positive_cases():
    for text in SHOULD_MATCH:
        assert is_webapp_request(text), f"expected a match for: {text!r}"
    print(f"[ok] all {len(SHOULD_MATCH)} website-style requests detected")


def check_negative_cases():
    for text in SHOULD_NOT_MATCH:
        assert not is_webapp_request(text), f"expected NO match for: {text!r}"
    print(f"[ok] all {len(SHOULD_NOT_MATCH)} plain script/function requests correctly ignored")


if __name__ == "__main__":
    check_positive_cases()
    check_negative_cases()
    print("\nAll webapp_detect checks passed.")
