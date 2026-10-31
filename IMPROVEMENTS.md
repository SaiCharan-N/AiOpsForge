# Improvement roadmap — status

Source: the five-item enhancement spec. Order of delivery: 1 → 3 → 2 → 4 → 5. **All five are implemented.**

| # | Improvement | Status | Where it lives | Tests |
|---|---|---|---|---|
| 1 | Context-aware failure recovery | ✅ done | `agents/failure_parser.py`, `agents/failure_analyzer.py`, `memory.py` (`failure_patterns`) | `scripts/test_failure_recovery.py` |
| 2 | Multi-turn user intervention (pause / feedback / resume) | ✅ done | `pipeline/intervention.py`, `/projects/{id}/intervention`, `/feedback`, `/interventions`, `components/PausePanel.jsx` | `scripts/test_intervention.py`, `frontend/tests/pause.test.jsx` |
| 3 | Cost & performance metrics | ✅ done | `pipeline/metrics.py`, `metrics_report.py`, `GET /metrics/*`, `pages/MetricsDash.jsx` | `scripts/test_metrics.py`, `frontend/tests/dash.test.jsx` |
| 4 | Framework-agnostic scaffolding | ✅ done | `scaffolds/`, `agents/scaffolder.py`, `webapp_detect.py`, `mcp_server/tools/check_webapp.py` | `scripts/test_frameworks.py`, `frontend/tests/framework.test.jsx` |
| 5 | Batch processing & comparative experiments | ✅ done | `batch.py`, `experiments.py`, `stats.py`, `run_config.py`, `/batch`, `/experiments`, `pages/BatchesPage.jsx`, `scripts/integrate_experiments.py` | `scripts/test_stats.py`, `scripts/test_batch.py`, `scripts/test_integrate_experiments.py`, `frontend/tests/batches.test.jsx` |

## Known gaps / decisions to make (not part of the five items)
- Nothing here has been run against the real Docker + Ollama + Postgres stack — all tests are offline (fake Ollama, SQLite, real pytest/node/flask/django). First thing to do on your machine: `docker compose up --build`.
- No Groq routing exists in this code base (`llm.py` is Ollama-only).
- Results from earlier memory-reuse experiments predate the `retrieve_similar` fix (#1) — re-run before citing them. `experiments/results/` is still simulated pilot data: **run the experiments on your hardware** (`python scripts/integrate_experiments.py <preset>`) before citing any number.
- Experiment arms run sequentially in the order given (no randomisation of arm order); per-run model overrides do not change the embedding model; default Ollama serialises generation, so latency under concurrency > 1 includes queueing.
- A paused run does not survive a backend restart (#2); a skipped task's leftover *test* file can fail later tasks' QA (pre-existing).
- A retry re-scaffold doesn't delete files from the previous attempt (pre-existing).
- The "30% fewer repeated errors" target of #1 is measurable (`task_history.repeated_errors`) but has not been measured.
- The UI was never viewed in a real browser (only jsdom).
