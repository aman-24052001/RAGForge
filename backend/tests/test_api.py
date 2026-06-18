"""
backend/tests/test_api.py — FastAPI integration tests (pure-Python fallback mode)
"""
import pytest
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Force Python fallback (no compiled .so needed in CI)
import sys
sys.modules.pop("ragforge_core", None)

from main import app

client = TestClient(app)


def test_status():
    r = client.get("/status")
    assert r.status_code == 200
    d = r.json()
    assert "total_chunks" in d
    assert "doc_ids" in d


def test_upload_txt():
    content = b"RAGForge is a retrieval system. It uses HNSW for fast search. BM25 handles lexical matching."
    r = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["chunks_indexed"] >= 1
    assert "test" in d["doc_id"]


def test_upload_unsupported():
    r = client.post(
        "/upload",
        files={"file": ("bad.csv", io.BytesIO(b"a,b,c"), "text/csv")},
    )
    assert r.status_code == 400


def test_query_no_docs():
    # With empty index
    r = client.post("/query", json={"query": "what is RAG?", "top_k": 5, "top_n": 3})
    assert r.status_code == 200
    d = r.json()
    assert "chunks" in d


def test_full_ingest_query_cycle():
    # Ingest
    text = (
        "Retrieval Augmented Generation combines neural retrieval with language models. "
        "HNSW is a graph-based approximate nearest neighbor algorithm. "
        "BM25 is a bag-of-words relevance ranking function used in information retrieval. "
        "MMR maximizes marginal relevance to ensure diversity in results."
    )
    client.post(
        "/upload",
        files={"file": ("rag_intro.txt", io.BytesIO(text.encode()), "text/plain")},
    )

    # Query
    r = client.post("/query", json={"query": "what is HNSW?", "top_k": 10, "top_n": 3})
    assert r.status_code == 200
    d = r.json()
    assert len(d["chunks"]) >= 1
    # Top chunk should mention HNSW
    top_text = d["chunks"][0]["chunk_text"].lower()
    assert "hnsw" in top_text or "retrieval" in top_text


def test_empty_query():
    r = client.post("/query", json={"query": "", "top_k": 10, "top_n": 3})
    assert r.status_code == 400


def test_clear_index():
    r = client.delete("/index")
    assert r.status_code == 200
