#!/usr/bin/env bash
# scripts/build.sh — full build pipeline
# Usage: ./scripts/build.sh [--skip-model] [--skip-cpp]
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$ROOT/models/minilm"
BUILD_DIR="$ROOT/rag_engine/build"

echo "══════════════════════════════════════════"
echo "  RAGForge Build Script"
echo "══════════════════════════════════════════"

# ── 1. Download MiniLM ONNX model ─────────────────────────────────────────────
if [[ "$*" != *"--skip-model"* ]]; then
  echo ""
  echo "▶ Step 1: Downloading all-MiniLM-L6-v2 ONNX model…"
  mkdir -p "$MODELS_DIR"

  if [ ! -f "$MODELS_DIR/model.onnx" ]; then
    pip install -q huggingface_hub optimum[onnxruntime] sentence-transformers

    python3 - <<'PYEOF'
import os, sys
from pathlib import Path
out = Path(os.environ.get("MODELS_DIR", "./models/minilm"))
out.mkdir(parents=True, exist_ok=True)

# Export to ONNX via optimum
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
print(f"  Exporting {model_id} to ONNX…")
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained(str(out))
tokenizer.save_pretrained(str(out))

# Rename to expected filename
import shutil
candidates = list(out.glob("*.onnx"))
if candidates and candidates[0].name != "model.onnx":
    shutil.move(str(candidates[0]), str(out / "model.onnx"))

print(f"  ✓ Model saved to {out}")
PYEOF
  else
    echo "  ✓ Model already downloaded"
  fi
fi

# ── 2. Install ONNX Runtime C++ headers ───────────────────────────────────────
if [[ "$*" != *"--skip-cpp"* ]]; then
  echo ""
  echo "▶ Step 2: Setting up ONNX Runtime C++ library…"

  ORT_VERSION="1.18.0"
  ORT_PLATFORM="linux-x64"   # Change for Mac: osx-x86_64 or osx-arm64
  ORT_DIR="/usr/local/onnxruntime"

  if [ ! -d "$ORT_DIR" ]; then
    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.tgz"
    echo "  Downloading ONNX Runtime $ORT_VERSION…"
    cd /tmp
    curl -L "$ORT_URL" -o ort.tgz
    tar xzf ort.tgz
    sudo mv "onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}" "$ORT_DIR"
    echo "  ✓ ONNX Runtime installed to $ORT_DIR"
  else
    echo "  ✓ ONNX Runtime already installed"
  fi

  # ── 3. Build C++ engine ─────────────────────────────────────────────────────
  echo ""
  echo "▶ Step 3: Building C++ RAG engine…"

  pip install -q pybind11

  PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"

  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$PYBIND11_DIR" \
    -DONNXRUNTIME_ROOT="$ORT_DIR" \
    2>&1 | tail -5

  make -j$(nproc) ragforge_core
  echo "  ✓ ragforge_core.so built → backend/"

  # Run C++ tests
  echo ""
  echo "▶ Step 3b: Running C++ unit tests…"
  make -j$(nproc) test_chunker test_mmr
  ./test_chunker
  ./test_mmr
fi

# ── 4. Install Python deps ─────────────────────────────────────────────────────
echo ""
echo "▶ Step 4: Installing Python dependencies…"
pip install -q -r "$ROOT/backend/requirements.txt"
echo "  ✓ Python deps installed"

echo ""
echo "══════════════════════════════════════════"
echo "  ✓ Build complete!"
echo ""
echo "  Run: ./scripts/run.sh"
echo "  Or:  cd backend && uvicorn main:app --reload"
echo "══════════════════════════════════════════"
