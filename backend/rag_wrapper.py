"""
RAGWrapper — thin Python layer over the C++ pybind11 ragforge_core module.
Falls back to a pure-Python implementation using sentence-transformers + numpy
if the compiled .so is not available (useful for local dev without a build).
"""

import os
import logging
from typing import Any

logger = logging.getLogger("ragforge.wrapper")

# ── Try loading the C++ module ─────────────────────────────────────────────────
try:
    import ragforge_core  # compiled pybind11 .so
    _CPP_AVAILABLE = True
    logger.info("Using C++ ragforge_core module")
except ImportError:
    _CPP_AVAILABLE = False
    logger.warning("ragforge_core.so not found — falling back to Python RAG implementation")


class RAGWrapper:
    """
    Unified interface regardless of backend (C++ or Python fallback).
    All methods return plain Python dicts/lists for easy JSON serialization.
    """

    def __init__(self):
        model_path = os.getenv("EMBEDDING_MODEL_PATH", "./models/minilm")
        chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
        mmr_lambda = float(os.getenv("MMR_LAMBDA", "0.6"))

        if _CPP_AVAILABLE:
            self._engine = ragforge_core.RAGEngine(
                model_path,
                chunk_size,
                chunk_overlap,
                mmr_lambda,
            )
            self._backend = "cpp"
        else:
            self._engine = _PythonRAGEngine(chunk_size, chunk_overlap, mmr_lambda)
            self._backend = "python"

    def ingest(self, text: str, doc_id: str) -> int:
        return self._engine.ingest(text, doc_id)

    def query(self, query_text: str, top_k: int = 20, top_n: int = 5) -> list[dict]:
        results = self._engine.query(query_text, top_k, top_n)
        if self._backend == "cpp":
            return [
                {
                    "chunk_text":      r.chunk_text,
                    "doc_id":          r.doc_id,
                    "chunk_index":     r.chunk_index,
                    "relevance_score": r.relevance_score,
                    "diversity_rank":  r.diversity_rank,
                }
                for r in results
            ]
        return results  # already dicts from Python fallback

    def save(self, path: str):
        self._engine.save_index(path)

    def load(self, path: str):
        self._engine.load_index(path)

    def stats(self) -> dict:
        s = self._engine.stats()
        if self._backend == "cpp":
            return {"total_chunks": s.total_chunks, "doc_ids": s.doc_ids}
        return s


# ── Pure-Python fallback (no build required) ───────────────────────────────────

class _PythonRAGEngine:
    """
    Fallback RAG using sentence-transformers + numpy.
    Same interface as C++ RAGEngine. Slower but requires no build step.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int, mmr_lambda: float):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mmr_lambda = mmr_lambda
        self._chunks: list[dict] = []
        self._embeddings: list[Any] = []
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _chunk_text(self, text: str, doc_id: str) -> list[dict]:
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append({
                "text": " ".join(chunk_words),
                "doc_id": doc_id,
                "chunk_index": len(chunks),
            })
        return chunks

    def ingest(self, text: str, doc_id: str) -> int:
        import numpy as np
        chunks = self._chunk_text(text, doc_id)
        model = self._get_model()
        embs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
        for c, e in zip(chunks, embs):
            c["embedding"] = e
            self._chunks.append(c)
            self._embeddings.append(e)
        return len(chunks)

    def query(self, query_text: str, top_k: int = 20, top_n: int = 5) -> list[dict]:
        import numpy as np
        if not self._chunks:
            return []
        model = self._get_model()
        q_emb = model.encode([query_text], normalize_embeddings=True)[0]

        emb_matrix = np.array(self._embeddings)
        scores = emb_matrix @ q_emb

        top_indices = scores.argsort()[::-1][:top_k]
        candidates = [
            {"chunk": self._chunks[i], "score": float(scores[i])}
            for i in top_indices
        ]

        # MMR
        selected = []
        remaining = list(candidates)
        for _ in range(min(top_n, len(remaining))):
            if not remaining:
                break
            best, best_score = None, -1e9
            for c in remaining:
                rel = self.mmr_lambda * c["score"]
                red = 0.0
                if selected:
                    sel_embs = np.array([s["chunk"]["embedding"] for s in selected])
                    sims = sel_embs @ c["chunk"]["embedding"]
                    red = (1.0 - self.mmr_lambda) * float(sims.max())
                mmr = rel - red
                if mmr > best_score:
                    best_score, best = mmr, c
            selected.append(best)
            remaining.remove(best)

        return [
            {
                "chunk_text":      s["chunk"]["text"],
                "doc_id":          s["chunk"]["doc_id"],
                "chunk_index":     s["chunk"]["chunk_index"],
                "relevance_score": s["score"],
                "diversity_rank":  float(i),
            }
            for i, s in enumerate(selected)
        ]

    def save_index(self, path: str):
        import json, numpy as np
        data = [
            {k: v.tolist() if hasattr(v, "tolist") else v for k, v in c.items()}
            for c in self._chunks
        ]
        with open(path + ".py.json", "w") as f:
            json.dump(data, f)

    def load_index(self, path: str):
        import json, numpy as np
        with open(path + ".py.json") as f:
            data = json.load(f)
        self._chunks = data
        self._embeddings = [np.array(c["embedding"]) for c in data]

    def stats(self) -> dict:
        doc_ids = list({c["doc_id"] for c in self._chunks})
        return {"total_chunks": len(self._chunks), "doc_ids": doc_ids}
