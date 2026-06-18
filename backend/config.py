"""Central config — all env-driven, with safe defaults."""
import os

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models/minilm/model.onnx")
INDEX_PATH           = os.getenv("INDEX_PATH", "./index/ragforge")
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", "64"))
MMR_LAMBDA           = float(os.getenv("MMR_LAMBDA", "0.6"))

# In CI / test mode: use tiny model name so sentence-transformers downloads fast
CI_MODE              = os.getenv("CI", "false").lower() == "true"
FALLBACK_MODEL_NAME  = "all-MiniLM-L6-v2" if not CI_MODE else "all-MiniLM-L6-v2"
