"""
backend/tests/test_api.py
Integration tests — uses real C++ .so if available, mock embedder otherwise.
"""
import sys, io, os
import pytest
import numpy as np

sys.modules.pop("ragforge_core", None)


class _MockST:
    """Deterministic unit-norm embeddings — no network, no model download."""
    def encode(self, texts, normalize_embeddings=True):
        rng = np.random.default_rng(42)
        embs = rng.random((len(texts), 384)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / (norms + 1e-9)


def _patch_model():
    """Patch _get_model in rag_wrapper to return mock — skips HuggingFace download."""
    import rag_wrapper
    rag_wrapper._model = _MockST()


@pytest.fixture(scope="module")
def client():
    _patch_model()
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as c:
        yield c


def test_status(client):
    r = client.get("/status")
    assert r.status_code == 200
    d = r.json()
    assert "total_chunks" in d
    assert "doc_ids" in d
    assert "llm_available" in d


def test_upload_txt(client):
    content = (
        b"RAGForge is a retrieval system. "
        b"It uses HNSW for fast approximate nearest neighbor search. "
        b"BM25 handles lexical matching alongside vector search. "
        b"MMR ensures diversity in retrieved results."
    )
    r = client.post("/upload",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")})
    assert r.status_code == 200
    d = r.json()
    assert d["chunks_indexed"] >= 1
    assert d["doc_id"]


def test_upload_unsupported(client):
    r = client.post("/upload",
        files={"file": ("data.csv", io.BytesIO(b"a,b"), "text/csv")})
    assert r.status_code == 400


def test_upload_empty(client):
    r = client.post("/upload",
        files={"file": ("empty.txt", io.BytesIO(b"   "), "text/plain")})
    assert r.status_code == 422


def test_empty_query_rejected(client):
    r = client.post("/query", json={"query": "", "top_k": 5, "top_n": 3})
    assert r.status_code == 400


def test_ingest_and_query(client):
    text = (
        b"HNSW stands for Hierarchical Navigable Small World. "
        b"It builds a multi-layer proximity graph for ANN search. "
        b"BM25 is a probabilistic lexical ranking function. "
        b"MMR maximizes marginal relevance for diverse retrieval."
    )
    up = client.post("/upload",
        files={"file": ("rag.txt", io.BytesIO(text), "text/plain")})
    assert up.status_code == 200

    r = client.post("/query", json={"query": "what is HNSW?", "top_k": 10, "top_n": 3})
    assert r.status_code == 200
    d = r.json()
    assert len(d["chunks"]) >= 1
    assert d["llm_answer"] is None   # no API key in CI
    for chunk in d["chunks"]:
        assert 0.0 <= chunk["relevance_score"] <= 1.0
        assert "chunk_text" in chunk
        assert "doc_id" in chunk


def test_clear_index(client):
    r = client.delete("/index")
    assert r.status_code == 200
    s = client.get("/status")
    assert s.json()["total_chunks"] == 0
