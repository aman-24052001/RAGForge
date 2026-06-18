"""
backend/tests/test_api.py — mocks fastembed so no model download in CI.
"""
import sys, io, os
import pytest
import numpy as np

sys.modules.pop("ragforge_core", None)


class _MockFastEmbed:
    """Returns deterministic unit-norm embeddings via fastembed's .embed() generator."""
    def embed(self, texts):
        rng = np.random.default_rng(42)
        for _ in texts:
            e = rng.random(384).astype(np.float32)
            yield e / (np.linalg.norm(e) + 1e-9)


@pytest.fixture(scope="module")
def client():
    import rag_wrapper
    # Patch _model directly so _embed() uses mock
    rag_wrapper._model = _MockFastEmbed()
    rag_wrapper._CPP_AVAILABLE = False

    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as c:
        yield c


def test_status(client):
    r = client.get("/status")
    assert r.status_code == 200
    d = r.json()
    assert "total_chunks" in d and "doc_ids" in d

def test_upload_txt(client):
    content = b"RAGForge uses HNSW BM25 and MMR for retrieval. It is fast and accurate."
    r = client.post("/upload", files={"file": ("t.txt", io.BytesIO(content), "text/plain")})
    assert r.status_code == 200
    assert r.json()["chunks_indexed"] >= 1

def test_upload_bad_ext(client):
    r = client.post("/upload", files={"file": ("x.csv", io.BytesIO(b"a"), "text/csv")})
    assert r.status_code == 400

def test_empty_query(client):
    r = client.post("/query", json={"query": "", "top_k": 5, "top_n": 3})
    assert r.status_code == 400

def test_query_returns_chunks(client):
    txt = b"HNSW is a graph algorithm. BM25 is a lexical scorer. MMR adds diversity."
    client.post("/upload", files={"file": ("rag.txt", io.BytesIO(txt), "text/plain")})
    r = client.post("/query", json={"query": "what is HNSW?", "top_k": 10, "top_n": 3})
    assert r.status_code == 200
    d = r.json()
    assert len(d["chunks"]) >= 1
    for c in d["chunks"]:
        assert 0.0 <= c["relevance_score"] <= 1.0

def test_clear(client):
    r = client.delete("/index")
    assert r.status_code == 200
    assert client.get("/status").json()["total_chunks"] == 0
