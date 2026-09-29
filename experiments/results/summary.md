# Experiment Results Summary

## Experiment 1 — Memory-assisted speedup on exact repeats

| Task | First run (s / attempts) | Repeat run (s / attempts) | Memory used |
|---|---|---|---|
| a function that checks if a number is prime, with ... | 0.86s / 2 | 0.47s / 1 | True |
| a function that flattens a nested list, with a tes... | 0.79s / 2 | 0.42s / 1 | True |
| a function that removes duplicate items from a lis... | 0.86s / 2 | 0.46s / 1 | True |

Average time reduction on repeat runs: **46%**

## Experiment 2 — MCP dynamic tool-discovery overhead

- Tool discovery (one-time, per session): 0.012s for 3 tools
- Avg MCP `write_file` call: 0.01342s
- Avg direct (no-protocol) call: 4e-05s
- Overhead per call: 0.01338s

## Experiment 3 — Attempts trend across a bug pattern

| Position | Attempts | Memory used |
|---|---|---|
| 1 | 3 | False |
| 2 | 2 | False |
| 3 | 1 | True |
| 4 | 1 | True |
| 5 | 1 | True |

Average attempts, first half of sequence: 2.50
Average attempts, second half of sequence: 1.00