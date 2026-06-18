"""
backend/tests/test_api.py
Pure unit/integration tests — no model download, no network calls.
Uses a mock embedder so CI passes instantly.
"""
import sys
import io
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Remove compiled C++ module if present
sys.modules.pop("ragforge_core", None)


# ── Mock SentenceTransformer so no model downloads in CI ─────────────────────
class _MockST:
    """Returns random unit-norm embeddings."""
    def encode(self, texts, normalize_embeddings=True):
        rng = np.random.default_rng(42)
        embs = rng.random((len(texts), 384)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / norms


@pytest.fixture(scope="module")
def client():
    # Patch SentenceTransformer before any app import
    with patch("rag_wrapper.SentenceTransformer", return_value=_MockST(), create=True):
        # Also patch the import inside _PythonRAGEngine._get_model
        import rag_wrapper
        rag_wrapper._CPP_AVAILABLE = False  # force Python path

        # Monkeypatch _get_model to return mock
        original_get_model = rag_wrapper._PythonRAGEngine._get_model
        def _mock_get_model(self):
            if self._model is None:
                self._model = _MockST()
            return self._model
        rag_wrapper._PythonRAGEngine._get_model = _mock_get_model

        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app) as c:
            yield c

        rag_wrapper._PythonRAGEngine._get_model = original_get_model


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
    r = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["chunks_indexed"] >= 1
    assert d["doc_id"]
    assert d["filename"] == "test.txt"


def test_upload_unsupported_extension(client):
    r = client.post(
        "/upload",
        files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
    )
    assert r.status_code == 400


def test_upload_empty_file(client):
    r = client.post(
        "/upload",
        files={"file": ("empty.txt", io.BytesIO(b"   "), "text/plain")},
    )
    assert r.status_code == 422


def test_empty_query_rejected(client):
    r = client.post("/query", json={"query": "", "top_k": 5, "top_n": 3})
    assert r.status_code == 400


def test_query_returns_chunks(client):
    # Ingest first
    text = (
        b"HNSW stands for Hierarchical Navigable Small World. "
        b"It is used for approximate nearest neighbor search. "
        b"The algorithm builds a multi-layer proximity graph."
    )
    client.post("/upload", files={"file": ("hnsw.txt", io.BytesIO(text), "text/plain")})

    r = client.post("/query", json={"query": "what is HNSW?", "top_k": 10, "top_n": 3})
    assert r.status_code == 200
    d = r.json()
    assert "chunks" in d
    assert "llm_available" in d
    # No LLM key in CI
    assert d["llm_answer"] is None
    for chunk in d["chunks"]:
        assert "chunk_text" in chunk
        assert "doc_id" in chunk
        assert "relevance_score" in chunk
        assert 0.0 <= chunk["relevance_score"] <= 1.0


def test_clear_index(client):
    r = client.delete("/index")
    assert r.status_code == 200
