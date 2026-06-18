# RAGForge

**Production-grade RAG system** — C++ engine with Python FastAPI backend and a static frontend on GitHub Pages.

| Component | Technology |
|-----------|-----------|
| Chunker | Sentence-aware sliding window (C++) |
| Embedder | `all-MiniLM-L6-v2` via ONNX Runtime (C++) |
| Index | HNSW (`hnswlib`) (C++) |
| Retrieval | Hybrid ANN + BM25 (C++) |
| Diversity | MMR — Maximal Marginal Relevance (C++) |
| Python binding | `pybind11` |
| Backend | FastAPI + uvicorn |
| LLM synthesis | Optional — Anthropic Claude / OpenAI (auto-detected) |
| Frontend | Static HTML/JS → GitHub Pages |
| Backend hosting | Render free tier |

---

## RAG Pipeline

```
Query → Embed → ANN Search (HNSW) ──┐
                                     ├→ Hybrid Score (60% cosine + 40% BM25) → MMR → Top-N chunks
              BM25 Inverted Index ───┘                                           ↓
                                                                        [LLM if key set]
                                                                        else show chunks
```

---

## Quick Start

### 1. Download model
```bash
python scripts/download_model.py
```

### 2. Build C++ engine
```bash
./scripts/build.sh
```
> On macOS: change `ORT_PLATFORM="osx-arm64"` in `build.sh` for Apple Silicon.

### 3. Run backend
```bash
./scripts/run.sh
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Open frontend
Open `frontend/index.html` in a browser, or visit the GitHub Pages URL.

---

## Pure-Python fallback (no C++ build)

The backend automatically falls back to `sentence-transformers` + `numpy` if `ragforge_core.so` is not present. This works out-of-the-box for development and CI:

```bash
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --reload
```

---

## Optional: LLM synthesis

Set one of:
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # uses claude-haiku-4-5 (fastest)
export OPENAI_API_KEY=sk-...          # uses gpt-4o-mini
```
If neither is set, the API returns retrieved chunks only (default behavior).

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Index stats + LLM availability |
| `/upload` | POST (multipart) | Ingest PDF/TXT/MD/DOCX |
| `/query` | POST (JSON) | Retrieve + optional LLM answer |
| `/query/stream` | POST (JSON) | SSE streaming version |
| `/index` | DELETE | Clear index |

```json
// POST /query
{ "query": "what is HNSW?", "top_k": 20, "top_n": 5 }

// Response
{
  "query": "what is HNSW?",
  "chunks": [
    { "doc_id": "intro_rag", "chunk_index": 2, "chunk_text": "...", "relevance_score": 0.87, "diversity_rank": 0 }
  ],
  "llm_answer": "HNSW (Hierarchical Navigable Small World)...",  // null if no key
  "llm_available": true
}
```

---

## Deployment

### Frontend → GitHub Pages
Push to `main` — GitHub Actions auto-deploys `frontend/` to Pages.

### Backend → Render
1. Connect repo to [render.com](https://render.com)
2. It picks up `render.yaml` automatically
3. Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in Render's env dashboard (optional)
4. Update `API` constant in `frontend/index.html` to your Render URL

### Docker
```bash
docker build -t ragforge .
docker run -p 8000:8000 ragforge
```

---

## Project Structure

```
ragforge/
├── rag_engine/
│   ├── src/            # C++: chunker, embedder, index, mmr, rag_engine
│   ├── include/        # Headers
│   ├── bindings/       # pybind11 module
│   ├── tests/          # C++ unit tests
│   └── CMakeLists.txt
├── backend/
│   ├── main.py         # FastAPI app
│   ├── rag_wrapper.py  # C++/Python unified interface
│   ├── doc_parser.py   # PDF/DOCX/TXT parser
│   ├── llm_client.py   # Optional LLM synthesis
│   └── tests/          # Python integration tests
├── frontend/
│   └── index.html      # Single-file UI
├── models/minilm/      # ONNX model (downloaded via script)
├── scripts/
│   ├── build.sh
│   ├── run.sh
│   └── download_model.py
├── .github/workflows/build.yml
├── render.yaml
└── Dockerfile
```

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL_PATH` | `./models/minilm/model.onnx` | ONNX model path |
| `INDEX_PATH` | `./index/ragforge` | Index save/load path |
| `CHUNK_SIZE` | `512` | Chars per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `MMR_LAMBDA` | `0.6` | MMR trade-off (0=diversity, 1=relevance) |
| `ANTHROPIC_API_KEY` | — | Enables Claude synthesis |
| `OPENAI_API_KEY` | — | Enables GPT synthesis (fallback) |
