#!/usr/bin/env python3
"""Week 11, Day 54: turn the three experiments' raw JSON into the tables and
charts your paper's Results section needs.

Reads experiments/results/exp{1,2,3}_*.json (produced by running the
exp*.py scripts against your live stack) and writes:
    experiments/results/exp1_chart.png   — first vs repeat run time/attempts
    experiments/results/exp2_chart.png   — MCP call time vs direct baseline
    experiments/results/exp3_chart.png   — attempts trend across a bug sequence
    experiments/results/summary.md       — the numbers, in prose + tables

Usage:
    python3 experiments/analyze_results.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from lib import load_results, RESULTS_DIR


def chart_exp1(data: dict) -> str:
    pairs = data["pairs"]
    labels = [f"Task {i+1}" for i in range(len(pairs))]
    first_times = [p["first_run"]["time_s"] for p in pairs]
    repeat_times = [p["repeat_run"]["time_s"] for p in pairs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = range(len(labels))
    width = 0.35
    ax1.bar([i - width/2 for i in x], first_times, width, label="First run", color="#6366f1")
    ax1.bar([i + width/2 for i in x], repeat_times, width, label="Repeat run", color="#16a34a")
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("Wall-clock time (s)")
    ax1.set_title("Experiment 1: time, first vs. repeat run")
    ax1.legend()

    first_attempts = [p["first_run"]["attempts"] for p in pairs]
    repeat_attempts = [p["repeat_run"]["attempts"] for p in pairs]
    ax2.bar([i - width/2 for i in x], first_attempts, width, label="First run", color="#6366f1")
    ax2.bar([i + width/2 for i in x], repeat_attempts, width, label="Repeat run", color="#16a34a")
    ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Attempts to pass")
    ax2.set_title("Experiment 1: attempts, first vs. repeat run")
    ax2.legend()

    fig.tight_layout()
    out = RESULTS_DIR / "exp1_chart.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def chart_exp2(data: dict) -> str:
    mcp_times = data["mcp"]["call_times_s"]
    direct_times = data["direct_baseline"]["call_times_s"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([mcp_times, direct_times])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MCP (discovery + call)", "Direct call (baseline)"])
    ax.set_ylabel("Time per write_file call (s)")
    ax.set_title(f"Experiment 2: MCP overhead over {data['n_calls']} calls")
    fig.tight_layout()
    out = RESULTS_DIR / "exp2_chart.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def chart_exp3(data: dict) -> str:
    seq = data["sequence"]
    positions = [s["position_in_sequence"] for s in seq]
    attempts = [s["attempts"] for s in seq]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#c4b5fd" if s["memory_used"] else "#9ca3af" for s in seq]
    ax.bar(positions, attempts, color=colors)
    ax.plot(positions, attempts, color="#6366f1", marker="o", linewidth=1.5)
    ax.set_xlabel("Position in sequence (same bug PATTERN, different task each time)")
    ax.set_ylabel("Attempts to pass")
    ax.set_title("Experiment 3: attempts trend across a bug pattern")
    ax.set_xticks(positions)
    fig.tight_layout()
    out = RESULTS_DIR / "exp3_chart.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def write_summary(exp1, exp2, exp3) -> Path:
    lines = ["# Experiment Results Summary", ""]

    lines.append("## Experiment 1 — Memory-assisted speedup on exact repeats")
    lines.append("")
    lines.append("| Task | First run (s / attempts) | Repeat run (s / attempts) | Memory used |")
    lines.append("|---|---|---|---|")
    for p in exp1["pairs"]:
        lines.append(
            f"| {p['request'][:50]}... "
            f"| {p['first_run']['time_s']}s / {p['first_run']['attempts']} "
            f"| {p['repeat_run']['time_s']}s / {p['repeat_run']['attempts']} "
            f"| {p['repeat_run']['memory_used']} |"
        )
    avg_first = sum(p["first_run"]["time_s"] for p in exp1["pairs"]) / len(exp1["pairs"])
    avg_repeat = sum(p["repeat_run"]["time_s"] for p in exp1["pairs"]) / len(exp1["pairs"])
    speedup = (avg_first - avg_repeat) / avg_first * 100 if avg_first else 0
    lines.append("")
    lines.append(f"Average time reduction on repeat runs: **{speedup:.0f}%**")
    lines.append("")

    lines.append("## Experiment 2 — MCP dynamic tool-discovery overhead")
    lines.append("")
    lines.append(f"- Tool discovery (one-time, per session): {exp2['mcp']['discovery_time_s']}s "
                  f"for {exp2['mcp']['tools_found']} tools")
    lines.append(f"- Avg MCP `write_file` call: {exp2['mcp']['avg_call_time_s']}s")
    lines.append(f"- Avg direct (no-protocol) call: {exp2['direct_baseline']['avg_call_time_s']}s")
    lines.append(f"- Overhead per call: {exp2['overhead_s']}s")
    lines.append("")

    lines.append("## Experiment 3 — Attempts trend across a bug pattern")
    lines.append("")
    lines.append("| Position | Attempts | Memory used |")
    lines.append("|---|---|---|")
    for s in exp3["sequence"]:
        lines.append(f"| {s['position_in_sequence']} | {s['attempts']} | {s['memory_used']} |")
    first_half = exp3["sequence"][:len(exp3["sequence"])//2]
    second_half = exp3["sequence"][len(exp3["sequence"])//2:]
    avg_first_half = sum(s["attempts"] for s in first_half) / len(first_half) if first_half else 0
    avg_second_half = sum(s["attempts"] for s in second_half) / len(second_half) if second_half else 0
    lines.append("")
    lines.append(f"Average attempts, first half of sequence: {avg_first_half:.2f}")
    lines.append(f"Average attempts, second half of sequence: {avg_second_half:.2f}")

    out = RESULTS_DIR / "summary.md"
    out.write_text("\n".join(lines))
    return out


def main():
    exp1 = load_results("exp1_memory_speedup")
    exp2 = load_results("exp2_mcp_overhead")
    exp3 = load_results("exp3_repeat_bug_trend")

    print("Generating charts...")
    print(" -", chart_exp1(exp1))
    print(" -", chart_exp2(exp2))
    print(" -", chart_exp3(exp3))

    summary_path = write_summary(exp1, exp2, exp3)
    print(f"\nSummary written to {summary_path}")
    print(f"\n{'='*60}\n{summary_path.read_text()}\n{'='*60}")


if __name__ == "__main__":
    main()
