#!/usr/bin/env bash
# Day 2: sanity-check that Ollama answers a raw prompt, no app code involved.
set -euo pipefail

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b}"
HOST="${OLLAMA_HOST_URL:-http://localhost:11434}"

echo "Asking ${MODEL} a test question..."
curl -s "${HOST}/api/generate" \
  -d "{\"model\": \"${MODEL}\", \"prompt\": \"Say hello in exactly five words.\", \"stream\": false}" \
  | python3 -m json.tool
