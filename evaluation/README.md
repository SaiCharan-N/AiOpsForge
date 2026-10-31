# Recording results for the paper — step by step

Goal: numbers a reviewer can trust, showing where AIOpsForge is better than an
**SE-Blackboard-style baseline** (and than the bare model), on **local 3B and 7B models**.
Nothing here invents results: every script writes plain CSVs you can open in Excel.

## What you will end up with

| Thing | Where it comes from |
|---|---|
| Hidden-test pass rate per model × arm (the real correctness number) | `run_ladder.py`, `raw_baseline.py` |
| "False pass" rate (passed its own tests, wrong anyway) | same CSVs (`false_pass`) |
| Tokens, latency, attempts, cost per run | pipeline metrics → CSV |
| IFS (substring + strict), planner-stage and final-output | `ifs_*` columns, `GET /projects/{id}/ifs` |
| LaTeX tables + PNG/PDF figures + `SUMMARY.md` | `make_paper_assets.py` |
| Machine / model / version record | `environment.json` |

## The comparison ladder

| Arm | What it is | Question it answers |
|---|---|---|
| A0 | raw model, one call, no pipeline | does the pipeline add anything at all? |
| A1 | message passing, 1 attempt, no memory | **the baseline** (SE-Blackboard-style, inside our framework) |
| A2 | blackboard, 1 attempt | SE-Blackboard's own question: blackboard vs message passing |
| A3 | blackboard, 3 attempts, failure recovery | our QA loop + Failure Analyzer |
| A3x | A3 with failure recovery **off** | what the Failure Analyzer alone contributes |
| A4 | A3 + long-term memory (wiped before each repetition) | does memory help across tasks? |

Say "SE-Blackboard-**style** baseline" in the paper — it is not their original system re-run.

## 0. One-time setup

1. `.env`: copy `.env.example`, keep `OLLAMA_EXTRA_MODELS=qwen2.5-coder:3b`, set `OLLAMA_MEM_LIMIT` (10g default).
   **A 7B model needs ~5 GB+ of free RAM on top of everything else.** On a machine with 8 GB total RAM it will swap and be
   extremely slow: then test the 3B only, or use a bigger machine, and say so in the paper.
2. Optional but recommended for reproducibility: `OLLAMA_SEED=…`, `OLLAMA_NUM_CTX=8192` in `.env` (the ladder also sets
   temperature 0.2, `num_ctx` 8192 and seeds 1,2,3 itself, per run).
3. `docker compose up -d --build`, then check both models exist: `docker compose exec ollama ollama list`
   (pull one by hand with `docker compose exec ollama ollama pull qwen2.5-coder:3b`).
4. On the host: `pip install -r evaluation/requirements.txt`.
5. Offline self-checks (no Ollama needed): `python scripts/test_phase18.py` and `python scripts/test_evaluation.py`.

## 1. Smoke test (5–15 minutes) — do this before anything long

```
python evaluation/run_ladder.py --models qwen2.5-coder:3b --arms A1,A3 --tasks T01,T04 --reps 1 --run-name smoke
```
Open `results/smoke/*.csv`. Check: rows exist, `hidden_total` is filled, `status` is not all `error`.
**Time how long it took and divide by the number of runs** — that is your minutes-per-run; multiply for the real plan
(`python evaluation/run_ladder.py --dry-run` prints the run count).

## 2. Pilot, then the real run

```
python evaluation/capture_environment.py --models qwen2.5-coder:3b,qwen2.5-coder:7b --out results/run1/environment.json
python evaluation/raw_baseline.py --models qwen2.5-coder:3b,qwen2.5-coder:7b --tasks core --reps 3 --run-name run1
python evaluation/run_ladder.py   --models qwen2.5-coder:3b,qwen2.5-coder:7b --tasks core --reps 3 --run-name run1
```
`core` = the 8 tasks marked `"core": true` in `tasks.json` (T01–T05, T07, T08, T13), `all` = 20. Start with `core`; add `--tasks all` only if time allows.
Interrupted? Run the same command again — finished arms are skipped. Leave the laptop plugged in, sleep off.

## 3. Extra experiments with the built-in presets (each is its own table in the paper)

```
python scripts/integrate_experiments.py communication_mode --repetitions 3 --concurrency 1   # blackboard vs message passing (+ IFS)
python scripts/integrate_experiments.py retry_policy       --repetitions 3 --concurrency 1   # 1 / 3 / 5 attempts
python scripts/integrate_experiments.py memory_speedup                                       # cold vs warm
python scripts/integrate_experiments.py repeat_bug_trend                                     # does memory reduce repeated errors?
python scripts/integrate_experiments.py mcp_overhead                                         # cost of the MCP tool layer
```
Note: the built-in presets use short requests with **no technical entities**, so their IFS column is empty. For IFS use
the ladder (its tasks are written with backticked names/numbers on purpose) or write your own requests the same way.
Web-app capability (frameworks): in the UI run one request per framework (FastAPI, Flask, Django, Express, Next.js) on the 7B,
record pass/fail and take a screenshot of each generated app.

## 4. IFS — how it is recorded

* The Planner stage and every Developer attempt write an `ifs` row (`app/ifs.py: record_ifs`). `GET /projects/<id>/ifs`
  shows planner score, per-task developer score, strict score, and **which request entities were lost**.
* CSV columns: `ifs_planner`, `ifs_avg` (developer, last attempt), `ifs_strict_avg`; the evaluation adds `ifs_final`,
  `ifs_final_strict` (final output vs request, identical matcher for every arm — use these to compare arms).
* A request without technical entities has IFS = empty, never 0.
* "Strict" is **our** stricter matcher (whole token or difflib ≥ 0.85); do not call it SE-Blackboard's.

## 5. Build tables and figures

```
python evaluation/make_paper_assets.py --results results/run1
```
Output in `results/run1/paper_assets/`: `tables/*.tex|md|csv`, `figures/*.png|pdf`, `SUMMARY.md`, `stats.json`.
Read `SUMMARY.md` first: it says which claims the data supports and which it does not.

## 6. Screenshots to take (illustration only — numbers come from the CSVs)

1. Chat UI mid-run with the flow diagram (Planner → Developer → QA), and once during a retry with the Failure Analyzer lit.
2. A failure-analysis bubble (root cause, location, variable values) next to the raw pytest error.
3. A memory-hit bubble ("similar past fix found").
4. The pause panel at the QA-failure point, your guidance, and the resumed run (a short qualitative case study).
5. Metrics dashboard: tokens by agent, latency waterfall, per-model speed.
6. Batches → Experiments page for each preset experiment.
7. The five generated web apps open in a browser.
8. `docker compose ps` and `docker compose exec ollama ollama list`.

## 7. What you may claim (and what you may not)

* Claim an improvement only where `SUMMARY.md` marks it SUPPORTED. "Inconclusive" is an honest result; say so.
* Compare against A1/A0, not against numbers from the SE-Blackboard paper (different task, model, hardware).
* Report: models + quantisation, hardware, temperature, seeds, `num_ctx`, number of tasks and repetitions, the prompt of A0.
* Report negative results (for example blackboard not helping a 3B model whose context window is small).
* Always show **hidden-test** pass rate next to the pipeline's own pass rate.
* Do not use anything from `experiments/results/` (simulated).

## Troubleshooting

* **Many `error` rows**: backend/Ollama problem, not model failure — check `docker compose logs backend ollama`, rerun with `--force`.
* **7B never finishes / container restarts**: not enough memory (see step 0.1).
* **A4 stops at start**: the memory-wipe command failed; check `docker compose ps` (service `postgres`) or pass `--reset-memory-cmd`.
* **`hidden_status` = `import_error`/`missing_entry_point`**: the model produced code without the requested function name — a real failure.
* **Adding tasks**: edit `evaluation/task_defs.py` (write a reference solution — `python scripts/test_evaluation.py` proves your hidden
  tests accept it and reject a stub), then `python evaluation/task_defs.py` to regenerate `tasks.json`.
