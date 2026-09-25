#!/usr/bin/env python3
"""Experiment 2 (Week 11, Day 52): what does dynamic tool discovery cost?

Scoping note: building a full second pipeline with every tool hardcoded
(bypassing MCP entirely) is a much bigger lift than the other two
experiments for comparatively little insight — the *interesting* question
isn't "does the pipeline work without MCP" (of course it can), it's "what
is the actual overhead of discovering + calling a tool dynamically over
MCP, versus just calling the equivalent Python function directly?" This
script measures exactly that: real network/protocol round-trips through
the live MCP server, against an equivalent direct filesystem write with no
protocol involved.

Requires the backend's Python environment (for app.mcp_client) and a
running mcp-server. Run this from inside the backend container, or with
PYTHONPATH pointed at backend/ and MCP_SERVER_URL set to reach it.

Usage:
    cd backend && python3 ../experiments/exp2_mcp_overhead.py
"""
import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from lib import save_results  # noqa: E402

N_CALLS = 20


async def measure_mcp_discovery_and_calls() -> dict:
    from app.mcp_client import discover_tools, call_tool

    t0 = time.time()
    tools = await discover_tools()
    discovery_time = time.time() - t0

    call_times = []
    for i in range(N_CALLS):
        t0 = time.time()
        await call_tool("write_file", {
            "project_id": "exp2-overhead-test",
            "path": f"file_{i}.py",
            "content": "print('hello')",
        })
        call_times.append(time.time() - t0)

    return {
        "discovery_time_s": round(discovery_time, 4),
        "tools_found": len(tools),
        "call_times_s": [round(t, 4) for t in call_times],
        "avg_call_time_s": round(sum(call_times) / len(call_times), 4),
    }


def measure_direct_baseline() -> dict:
    call_times = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(N_CALLS):
            t0 = time.time()
            path = Path(tmpdir) / f"file_{i}.py"
            path.write_text("print('hello')")
            call_times.append(time.time() - t0)

    return {
        "call_times_s": [round(t, 5) for t in call_times],
        "avg_call_time_s": round(sum(call_times) / len(call_times), 5),
    }


def main():
    print(f"Measuring {N_CALLS} MCP write_file calls against the live mcp-server...")
    mcp_results = asyncio.run(measure_mcp_discovery_and_calls())
    print(f"  discovery: {mcp_results['discovery_time_s']}s for {mcp_results['tools_found']} tools")
    print(f"  avg call: {mcp_results['avg_call_time_s']}s\n")

    print(f"Measuring {N_CALLS} direct filesystem writes (no protocol) as baseline...")
    direct_results = measure_direct_baseline()
    print(f"  avg call: {direct_results['avg_call_time_s']}s\n")

    overhead_s = mcp_results["avg_call_time_s"] - direct_results["avg_call_time_s"]
    print(f"MCP protocol overhead per call: ~{overhead_s:.4f}s "
          f"({overhead_s / direct_results['avg_call_time_s']:.0f}x the direct-call baseline)")

    save_results("exp2_mcp_overhead", {
        "n_calls": N_CALLS,
        "mcp": mcp_results,
        "direct_baseline": direct_results,
        "overhead_s": round(overhead_s, 4),
    })


if __name__ == "__main__":
    main()
