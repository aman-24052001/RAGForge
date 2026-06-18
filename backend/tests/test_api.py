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

def test_login_no_password(client):
    # No APP_PASSWORD set → login returns no-auth token
    r = client.post("/auth/login", json={"password": "anything"})
    assert r.status_code == 200
    assert r.json()["token"] == "no-auth"

def test_status_open(client):
    r = client.get("/status")
    assert r.status_code == 200
    assert "total_chunks" in r.json()

def test_upload(client):
    r = client.post("/upload",
        files={"file": ("t.txt", io.BytesIO(b"HNSW BM25 MMR retrieval system test."), "text/plain")})
    assert r.status_code == 200
    assert r.json()["chunks_indexed"] >= 1

def test_query(client):
    client.post("/upload",
        files={"file": ("q.txt", io.BytesIO(b"HNSW is a graph algorithm for ANN search."), "text/plain")})
    r = client.post("/query", json={"query": "HNSW", "top_k": 5, "top_n": 3})
    assert r.status_code == 200
    assert len(r.json()["chunks"]) >= 1

def test_logout(client):
    r = client.post("/auth/logout")
    assert r.status_code == 200

def test_clear(client):
    r = client.delete("/index")
    assert r.status_code == 200
