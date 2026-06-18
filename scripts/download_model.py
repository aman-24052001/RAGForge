#!/usr/bin/env python3
"""
scripts/download_model.py
Downloads all-MiniLM-L6-v2 as ONNX + vocab.txt into models/minilm/
Run: python scripts/download_model.py
"""
import os
import sys
import shutil
from pathlib import Path

OUT = Path(__file__).parent.parent / "models" / "minilm"
OUT.mkdir(parents=True, exist_ok=True)

print("Installing export dependencies…")
os.system(f"{sys.executable} -m pip install -q optimum[onnxruntime] transformers sentencepiece")

print(f"\nExporting all-MiniLM-L6-v2 to ONNX → {OUT}")

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(str(OUT))
tokenizer.save_pretrained(str(OUT))

# Normalize to model.onnx
onnx_files = list(OUT.glob("*.onnx"))
if onnx_files and onnx_files[0].name != "model.onnx":
    shutil.move(str(onnx_files[0]), str(OUT / "model.onnx"))

print(f"\n✓ Files in {OUT}:")
for f in sorted(OUT.iterdir()):
    size = f.stat().st_size // 1024
    print(f"  {f.name:<30} {size:>6} KB")
print("\nDone. Set EMBEDDING_MODEL_PATH=" + str(OUT / "model.onnx"))
