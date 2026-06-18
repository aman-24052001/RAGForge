#!/usr/bin/env bash
# scripts/run.sh — start the RAGForge backend
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-$ROOT/models/minilm/model.onnx}"
export INDEX_PATH="${INDEX_PATH:-$ROOT/index/ragforge}"
export CHUNK_SIZE="${CHUNK_SIZE:-512}"
export CHUNK_OVERLAP="${CHUNK_OVERLAP:-64}"
export MMR_LAMBDA="${MMR_LAMBDA:-0.6}"

# Optional — set these for LLM synthesis:
# export ANTHROPIC_API_KEY=sk-ant-...
# export OPENAI_API_KEY=sk-...

echo "Starting RAGForge backend…"
echo "  Model:  $EMBEDDING_MODEL_PATH"
echo "  Index:  $INDEX_PATH"
echo "  Port:   8000"

cd "$ROOT/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
