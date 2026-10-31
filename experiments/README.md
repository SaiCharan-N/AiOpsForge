# experiments/

`exp1_memory_speedup.py`, `exp2_mcp_overhead.py` and `exp3_repeat_bug_trend.py` are the original standalone
scripts. They still work, but the experiments now live **inside the backend** (`/experiments`), which adds
concurrent runs, per-run settings, repetitions, statistics and exports. Prefer:

```bash
python scripts/integrate_experiments.py --list
python scripts/integrate_experiments.py memory_speedup      # exp1
python scripts/integrate_experiments.py mcp_overhead        # exp2
python scripts/integrate_experiments.py repeat_bug_trend    # exp3
python experiments/analyze_results.py                       # charts + summary.md, unchanged
```

`integrate_experiments.py` writes the legacy `exp{1,2,3}_*.json` files in the shape `analyze_results.py` reads,
plus the full result, the raw per-run CSV and LaTeX tables.

Note: the files currently in `results/` came from a simulated pilot (sub-second runs), not from your Ollama setup.
