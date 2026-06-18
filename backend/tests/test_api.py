"""
backend/tests/test_api.py — FastAPI integration tests (pure-Python fallback mode)
Runs in CI without the compiled .so — uses sentence-transformers fallback.
"""
import sys
import io
import pytest

# Ensure C++ module is NOT loaded so fallback activates
sys.modules.pop("ragforge_core", None)

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from main import app
    with TestClient(app) as c:
        yield c


def test_status(client):
    r = client.get("/status")
    assert r.status_code == 200
    d = r.json()
    assert "total_chunks" in d
    assert "doc_ids" in d


def test_upload_txt(client):
    content = (
        b"RAGForge is a retrieval system. "
        b"It uses HNSW for fast approximate nearest neighbor search. "
        b"BM25 handles lexical matching alongside vector search."
    )
    r = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["chunks_indexed"] >= 1
    assert d["doc_id"]


def test_upload_unsupported_extension(client):
    r = client.post(
        "/upload",
        files={"file": ("bad.csv", io.BytesIO(b"a,b,c"), "text/csv")},
    )
    assert r.status_code == 400


def test_empty_query_rejected(client):
    r = client.post("/query", json={"query": "", "top_k": 5, "top_n": 3})
    assert r.status_code == 400


def test_full_ingest_query_cycle(client):
    text = (
        "Retrieval Augmented Generation combines neural retrieval with language models. "
        "HNSW is a graph-based approximate nearest neighbor algorithm used in vector search. "
        "BM25 is a bag-of-words relevance ranking function used in information retrieval. "
        "MMR maximizes marginal relevance to ensure diversity in retrieved results."
    )
    upload = client.post(
        "/upload",
        files={"file": ("rag_intro.txt", io.BytesIO(text.encode()), "text/plain")},
    )
    assert upload.status_code == 200

    r = client.post("/query", json={"query": "what is HNSW?", "top_k": 10, "top_n": 3})
    assert r.status_code == 200
    d = r.json()
    assert len(d["chunks"]) >= 1
    # Scores should be between 0 and 1
    for chunk in d["chunks"]:
        assert 0.0 <= chunk["relevance_score"] <= 1.0


def test_clear_index(client):
    r = client.delete("/index")
    assert r.status_code == 200
