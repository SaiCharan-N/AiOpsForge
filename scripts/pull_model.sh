#!/usr/bin/env bash
# Day 2: pull the coding model into the running ollama container.
set -euo pipefail

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b}"

echo "Pulling model: ${MODEL} (this can take a while on first run)"
docker compose exec ollama ollama pull "${MODEL}"
echo "Done. Installed models:"
docker compose exec ollama ollama list
