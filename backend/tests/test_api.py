"""
Tests — covers open-access mode (no APP_PASSWORD) + auth flow with password.
"""
import sys, io, os
import pytest
import numpy as np

sys.modules.pop("ragforge_core", None)

class _MockFastEmbed:
    def embed(self, texts):
        rng = np.random.default_rng(42)
        for _ in texts:
            e = rng.random(384).astype(np.float32)
            yield e / (np.linalg.norm(e) + 1e-9)

@pytest.fixture(scope="module")
def client():
    import rag_wrapper
    rag_wrapper._model = _MockFastEmbed()
    rag_wrapper._CPP_AVAILABLE = False
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as c:
        yield c

# ── Open access tests (no APP_PASSWORD) ──────────────────────────────────────
def test_login_no_password(client):
    r = client.post("/auth/login", json={"password": "anything"})
    assert r.status_code == 200
    assert r.json()["token"] == "no-auth"

def test_status_open(client):
    r = client.get("/status")
    assert r.status_code == 200
    d = r.json()
    assert "total_chunks" in d
    assert "session_id" in d

def test_upload_txt(client):
    r = client.post("/upload",
        files={"file": ("t.txt", io.BytesIO(b"HNSW BM25 MMR retrieval system test sentence."), "text/plain")})
    assert r.status_code == 200
    assert r.json()["chunks_indexed"] >= 1

def test_upload_too_large(client):
    big = b"x" * (5 * 1024 * 1024 + 1)
    r = client.post("/upload",
        files={"file": ("big.txt", io.BytesIO(big), "text/plain")})
    assert r.status_code == 413

def test_upload_bad_ext(client):
    r = client.post("/upload",
        files={"file": ("x.csv", io.BytesIO(b"a,b"), "text/csv")})
    assert r.status_code == 400

def test_empty_query_rejected(client):
    r = client.post("/query", json={"query": "", "top_k": 5, "top_n": 3})
    assert r.status_code == 400

def test_query_returns_chunks(client):
    client.post("/upload",
        files={"file": ("rag.txt", io.BytesIO(b"HNSW is a graph algorithm. BM25 is lexical. MMR adds diversity."), "text/plain")})
    r = client.post("/query", json={"query": "HNSW", "top_k": 5, "top_n": 3})
    assert r.status_code == 200
    d = r.json()
    assert len(d["chunks"]) >= 1
    for c in d["chunks"]:
        assert isinstance(c["diversity_rank"], int)
        assert 0.0 <= c["relevance_score"] <= 1.0

def test_logout(client):
    r = client.post("/auth/logout")
    assert r.status_code == 200

def test_clear(client):
    r = client.delete("/index")
    assert r.status_code == 200
    assert client.get("/status").json()["total_chunks"] == 0

# ── Auth tests (with APP_PASSWORD set) ───────────────────────────────────────
def test_wrong_password():
    os.environ["APP_PASSWORD"] = "correct-horse"
    try:
        from fastapi.testclient import TestClient
        import rag_wrapper
        rag_wrapper._model = _MockFastEmbed()
        rag_wrapper._CPP_AVAILABLE = False
        from main import app
        with TestClient(app) as c:
            r = c.post("/auth/login", json={"password": "wrong"})
            assert r.status_code == 401

            r = c.get("/status")
            assert r.status_code == 401

            r = c.post("/auth/login", json={"password": "correct-horse"})
            assert r.status_code == 200
            token = r.json()["token"]
            assert "." in token  # session_id.signature

            r = c.get("/status", headers={"Authorization": f"Bearer {token}"})
            assert r.status_code == 200
    finally:
        del os.environ["APP_PASSWORD"]
