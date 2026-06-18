# RAGForge

> Production-grade RAG system with a C++ core, Python backend, and a neo-brutalist mobile-first UI.

**Live demo:** [aman-24052001.github.io/RAGForge](https://aman-24052001.github.io/RAGForge/)  
**Backend:** [ragforge-wd3z.onrender.com](https://ragforge-wd3z.onrender.com/docs)

---

## What it does

Upload documents (PDF, TXT, MD, DOCX), ask questions, get back the most relevant and diverse chunks — ranked, scored, and optionally synthesized by an LLM.

Every retrieval runs a full pipeline:

```
Query
  └─ Embed (fastembed / all-MiniLM-L6-v2)
       └─ ANN Search (HNSW — C++)
            └─ BM25 Hybrid Scoring (C++)
                 └─ MMR Diversity Reranking (C++)
                      └─ Top-N Chunks → [LLM synthesis if key set]
```

---

## Architecture

| Layer | Technology | Notes |
|-------|-----------|-------|
| Chunker | Sentence-aware sliding window | C++, no external deps |
| Embedding | `all-MiniLM-L6-v2` via `fastembed` | Python, ONNX Runtime, ~50MB |
| Vector Index | HNSW (`hnswlib` vendored) | C++, sub-ms search |
| Lexical Index | BM25 inverted index | C++, pure implementation |
| Hybrid Score | 60% cosine + 40% BM25 | C++ |
| Diversity | MMR (Maximal Marginal Relevance) | C++, λ=0.6 |
| Python Binding | `pybind11` | `.so` compiled in CI |
| Backend | FastAPI + uvicorn | Per-session isolated indexes |
| Auth | HMAC-SHA256 tokens | `APP_PASSWORD` env var |
| LLM (optional) | Anthropic Claude Haiku / GPT-4o-mini | Auto-detected from env |
| Frontend | Single-file HTML/JS | Neo-brutalist, mobile-first |
| Frontend hosting | GitHub Pages | Auto-deployed via Actions |
| Backend hosting | Render free tier | Docker, 512MB RAM |

---

## RAG Pipeline — deep dive

### Chunking
Sentence-aware sliding window with configurable size (default 512 chars) and overlap (64 chars). Splits on `.`, `!`, `?`, `\n\n` boundaries.

### Hybrid Retrieval
Two signals fused per candidate chunk:
- **Cosine similarity** via HNSW approximate nearest neighbour (over-fetches 2× top_k)
- **BM25** from a term-frequency inverted index built at ingest time

Combined as: `final_score = 0.6 × cosine + 0.4 × BM25_normalized`

### MMR Diversity
Maximal Marginal Relevance iteratively selects chunks that are relevant to the query but dissimilar to already-selected chunks:

```
MMR(d) = λ × relevance(d, query) − (1−λ) × max_similarity(d, selected)
```

λ=0.6 balances relevance vs diversity. Tunable via `MMR_LAMBDA` env var.

---

## Security

- **Password-protected** — set `APP_PASSWORD` on Render, never in code
- **Per-session isolation** — each login gets a unique `session_id`; 4 users = 4 independent indexes, zero data leakage
- **Token format** — `session_id.HMAC-SHA256(session_id:day)`, expires daily
- **Logout** — `POST /auth/logout` immediately wipes the session's index from server memory; frontend clears token from `sessionStorage`

---

## Running locally

### Prerequisites
- Python 3.11+
- `cmake`, `build-essential` (for C++ build)
- `pybind11` (`pip install pybind11`)

### 1. Build C++ engine (optional — Python fallback works without it)
```bash
cd rag_engine && mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
make -j$(nproc)
# Outputs: backend/ragforge_core*.so
```

### 2. Install Python deps
```bash
pip install -r backend/requirements.txt
```

### 3. Run backend
```bash
cd backend
uvicorn main:app --reload
# API at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 4. Open frontend
Open `frontend/index.html` in a browser, or point it at localhost:8000.

---

## Deployment

### Frontend → GitHub Pages (automatic)
Push to `main` → GitHub Actions builds C++, runs Python tests, deploys `frontend/` to Pages.

### Backend → Render (Docker)
1. Connect repo to [render.com](https://render.com) → New Web Service
2. Render detects `Dockerfile` automatically
3. Set env vars in Render dashboard:

| Variable | Required | Description |
|----------|----------|-------------|
| `APP_PASSWORD` | Yes | Shared access password (set in dashboard, never in code) |
| `ANTHROPIC_API_KEY` | Optional | Enables Claude Haiku LLM synthesis |
| `OPENAI_API_KEY` | Optional | Enables GPT-4o-mini synthesis (fallback) |
| `CHUNK_SIZE` | Optional | Default: 512 |
| `MMR_LAMBDA` | Optional | Default: 0.6 |

---

## API Reference

All endpoints (except `/auth/login`) require `Authorization: Bearer <token>`.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/login` | POST | — | Get session token |
| `/auth/logout` | POST | ✓ | Wipe session + index |
| `/status` | GET | ✓ | Chunk count, doc list, LLM availability |
| `/upload` | POST | ✓ | Ingest PDF/TXT/MD/DOCX |
| `/query` | POST | ✓ | Retrieve chunks + optional LLM answer |
| `/index` | DELETE | ✓ | Clear session index |

```bash
# Login
curl -X POST https://ragforge-wd3z.onrender.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"password": "your-password"}'
# → { "token": "abc123.hmac...", "session_id": "abc123" }

# Upload
curl -X POST https://ragforge-wd3z.onrender.com/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@document.pdf"

# Query
curl -X POST https://ragforge-wd3z.onrender.com/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "what is HNSW?", "top_k": 20, "top_n": 5}'
```

---

## Project structure

```
ragforge/
├── rag_engine/
│   ├── src/
│   │   ├── chunker.cpp          # Sentence-aware sliding window
│   │   ├── index.cpp            # HNSW + BM25 hybrid index
│   │   ├── mmr.cpp              # Maximal Marginal Relevance
│   │   └── rag_engine_core.cpp  # Unified entry point
│   ├── include/                 # C++ headers
│   ├── bindings/
│   │   └── rag_bindings.cpp     # pybind11 module
│   ├── third_party/
│   │   ├── hnswlib/             # Vendored — no FetchContent
│   │   └── nlohmann/            # Vendored json.hpp
│   ├── tests/                   # C++ unit tests
│   └── CMakeLists.txt
├── backend/
│   ├── main.py                  # FastAPI, per-session routing
│   ├── auth.py                  # HMAC token auth + session management
│   ├── rag_wrapper.py           # C++/Python unified interface
│   ├── doc_parser.py            # PDF/DOCX/TXT/MD → plain text
│   ├── llm_client.py            # Optional Anthropic/OpenAI synthesis
│   ├── requirements.txt
│   └── tests/
├── frontend/
│   └── index.html               # Single-file neo-brutalist UI
├── .github/workflows/
│   └── build.yml                # CI: C++ build → Python tests → Pages deploy
├── Dockerfile                   # For Render
├── render.yaml
└── README.md
```

---

## Notes on free tier limits

Render free tier spins down after **15 min of inactivity** — first request after idle takes ~30–50s to wake up. The frontend handles this with a retry loop showing "waking up…".

All session data (indexes, chunks) lives in memory. A restart wipes everything — users need to re-upload. This is a free tier constraint; persistent storage requires a paid plan or external DB.
