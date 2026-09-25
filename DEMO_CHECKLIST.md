# AIOpsForge — Demo Day Checklist (Days 59–60)

## The day before

- [ ] `docker compose down -v && docker compose up -d --build` from a totally clean checkout — catches "works on my machine" problems before the real demo does.
- [ ] `bash scripts/pull_model.sh` and confirm the model is actually pulled (`docker compose exec ollama ollama list`).
- [ ] `docker compose exec ollama ollama pull nomic-embed-text` (the embedding model — separate from your coding model).
- [ ] Run through the full script below once, start to finish, with a timer.
- [ ] Charge your laptop. Turn off notifications. Close Slack/email.

## The live demo script (aim for 8–10 minutes)

1. **Open the dashboard** (`http://localhost:5173`). One sentence on what it is: "Planner, Developer, and QA agents, coordinated by LangGraph, with tools discovered dynamically over MCP and a persistent memory layer."

2. **Submit a fresh request** — something with 2–3 tasks, e.g. *"a function that checks if a string is a palindrome, with a test, and a function that counts vowels in a string, with a test."* Narrate while it runs: "Planner just broke that into two tasks. Developer's writing task 1 now — that write is happening through an MCP tool call, not a hardcoded file write."

3. **Point at a pass.** "QA just ran pytest on that through MCP too, and it passed first try."

4. **Submit the SAME request again** (or a close variant). Point at the 🧠 badge: "That's long-term memory — the Developer just reused the fix from 30 seconds ago instead of solving it from scratch. That's not the same project either — this is a brand new project ID, on purpose, to show the memory isn't just an in-memory cache for one run."

5. **Show a deliberate failure** (optional, if time allows) — submit something oddly worded to increase the odds of a first-attempt failure, or reference the automated retry test results from Phase 2's README if live retries don't reliably reproduce on demand. Narrate: "If QA fails, the error gets fed back to the Developer automatically, up to 3 attempts, before it escalates for human review instead of looping forever."

6. **Download the project.** Click the download button, open the zip, show the actual generated files + tests.

7. **Close with the numbers** (from `experiments/results/summary.md`, or your own re-run numbers if you have them by demo day): memory reuse cut repeat-task time by ~46% and attempts from 2 to 1 in the pilot run; MCP protocol overhead was small relative to LLM latency.

## Fallback plan if live demo breaks

- Have `experiments/results/summary.md` and the three chart PNGs open in a separate tab/window, ready to show as "here's what a completed run produced" if live generation stalls or Ollama is slow on the day.
- Have a terminal ready with `python3 scripts/test_pipeline.py "..."` as a plain-text fallback if the dashboard itself has a rendering issue — the pipeline working matters more than the UI working.
- Know your one-liner for "why does this take 30–60 seconds per task": local LLM inference on non-datacenter hardware, not a system design flaw.

## Bug-bash checklist (do this BEFORE the day-before rehearsal, not during)

- [ ] `docker compose logs -f` while running one full request — skim for anything alarming, even non-fatal.
- [ ] Try a genuinely malformed request (empty string, or single word) — confirm it degrades gracefully instead of crashing.
- [ ] Kill and restart the `ollama` container mid-run once, on purpose — confirm the backend's `/health/deep` correctly reports it, and a subsequent request works once Ollama's back.
- [ ] Check `docker compose exec postgres psql -U aiops -d aiopsforge -c "SELECT status, count(*) FROM projects GROUP BY status;"` — nothing should be permanently stuck in `building`.
