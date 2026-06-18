"""
rag_wrapper.py — unified interface over C++ ragforge_core (pybind11) or Python fallback.

Architecture:
  C++ handles: chunking, HNSW indexing, BM25, hybrid scoring, MMR
  Python handles: embedding (sentence-transformers, no C++ ORT dependency)
"""

import os
import logging
import numpy as np
from typing import Any

logger = logging.getLogger("ragforge.wrapper")

try:
    import ragforge_core
    _CPP_AVAILABLE = True
    logger.info("Using C++ ragforge_core module")
except ImportError:
    _CPP_AVAILABLE = False
    logger.warning("ragforge_core.so not found — using pure-Python fallback")

# Shared embedding model (loaded once)
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

EMB_DIM = 384  # all-MiniLM-L6-v2


class RAGWrapper:
    def __init__(self):
        chunk_size    = int(os.getenv("CHUNK_SIZE",    "512"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "64"))
        mmr_lambda    = float(os.getenv("MMR_LAMBDA",  "0.6"))

        if _CPP_AVAILABLE:
            self._engine  = ragforge_core.RAGEngineCore(
                EMB_DIM, chunk_size, chunk_overlap, mmr_lambda)
            self._backend = "cpp"
        else:
            self._engine  = _PythonRAGEngine(chunk_size, chunk_overlap, mmr_lambda)
            self._backend = "python"

    def ingest(self, text: str, doc_id: str) -> int:
        model = _get_model()

        if self._backend == "cpp":
            # C++ chunks, Python embeds, C++ indexes
            chunk_texts = self._engine.chunk_text(text, doc_id)
            if not chunk_texts:
                return 0
            embeddings = model.encode(chunk_texts, normalize_embeddings=True)
            doc_ids = [doc_id] * len(chunk_texts)
            indices = list(range(len(chunk_texts)))
            emb_list = [e.tolist() for e in embeddings]
            self._engine.add_chunks(chunk_texts, doc_ids, indices, emb_list)
            return len(chunk_texts)
        else:
            return self._engine.ingest(text, doc_id, model)

    def query(self, query_text: str, top_k: int = 20, top_n: int = 5) -> list[dict]:
        model = _get_model()

        if self._backend == "cpp":
            q_emb = model.encode([query_text], normalize_embeddings=True)[0].tolist()
            results = self._engine.query(q_emb, query_text, top_k, top_n)
            return [
                {
                    "chunk_text":      r.chunk_text,
                    "doc_id":          r.doc_id,
                    "chunk_index":     r.chunk_index,
                    "relevance_score": float(r.relevance_score),
                    "diversity_rank":  float(r.diversity_rank),
                }
                for r in results
            ]
        else:
            return self._engine.query(query_text, top_k, top_n, model)

    def save(self, path: str):
        self._engine.save(path) if self._backend == "cpp" else self._engine.save_index(path)

    def load(self, path: str):
        self._engine.load(path) if self._backend == "cpp" else self._engine.load_index(path)

    def clear(self):
        self._engine.clear()

    def stats(self) -> dict:
        s = self._engine.stats()
        if self._backend == "cpp":
            return {"total_chunks": s.total_chunks, "doc_ids": list(s.doc_ids)}
        return s


# ── Pure-Python fallback ───────────────────────────────────────────────────────

class _PythonRAGEngine:
    def __init__(self, chunk_size: int, chunk_overlap: int, mmr_lambda: float):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mmr_lambda    = mmr_lambda
        self._chunks: list[dict] = []
        self._embeddings: list[Any] = []

    def _chunk_text(self, text: str, doc_id: str) -> list[dict]:
        words = text.split()
        step  = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append({
                "text":        " ".join(chunk_words),
                "doc_id":      doc_id,
                "chunk_index": len(chunks),
            })
        return chunks

    def ingest(self, text: str, doc_id: str, model) -> int:
        chunks = self._chunk_text(text, doc_id)
        embs   = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
        for c, e in zip(chunks, embs):
            c["embedding"] = e
            self._chunks.append(c)
            self._embeddings.append(e)
        return len(chunks)

    def query(self, query_text: str, top_k: int, top_n: int, model) -> list[dict]:
        if not self._chunks:
            return []
        q_emb      = model.encode([query_text], normalize_embeddings=True)[0]
        emb_matrix = np.array(self._embeddings)
        scores     = emb_matrix @ q_emb

        top_idx    = scores.argsort()[::-1][:top_k]
        candidates = [{"chunk": self._chunks[i], "score": float(scores[i])} for i in top_idx]

        # MMR
        selected, remaining = [], list(candidates)
        for _ in range(min(top_n, len(remaining))):
            if not remaining:
                break
            best, best_score = None, -1e9
            for c in remaining:
                rel = self.mmr_lambda * c["score"]
                red = 0.0
                if selected:
                    sel_embs = np.array([s["chunk"]["embedding"] for s in selected])
                    red = (1.0 - self.mmr_lambda) * float((sel_embs @ c["chunk"]["embedding"]).max())
                if rel - red > best_score:
                    best_score, best = rel - red, c
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
        import json
        data = [
            {k: v.tolist() if hasattr(v, "tolist") else v for k, v in c.items()}
            for c in self._chunks
        ]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path + ".py.json", "w") as f:
            json.dump(data, f)

    def load_index(self, path: str):
        import json
        with open(path + ".py.json") as f:
            data = json.load(f)
        self._chunks     = data
        self._embeddings = [np.array(c["embedding"]) for c in data]

    def clear(self):
        self._chunks.clear()
        self._embeddings.clear()

    def stats(self) -> dict:
        return {
            "total_chunks": len(self._chunks),
            "doc_ids": list({c["doc_id"] for c in self._chunks}),
        }
