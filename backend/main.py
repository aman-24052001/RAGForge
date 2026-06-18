"""
RAGForge Backend — FastAPI
Handles: document upload, indexing, querying, optional LLM synthesis
"""

import os
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from doc_parser import parse_document
from rag_wrapper import RAGWrapper
from llm_client import stream_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragforge")

# ── Global RAG instance ────────────────────────────────────────────────────────
rag: RAGWrapper | None = None
INDEX_PATH = Path(os.getenv("INDEX_PATH", "./index/ragforge"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    rag = RAGWrapper()
    if (INDEX_PATH.parent / (INDEX_PATH.name + ".meta.json")).exists():
        logger.info("Loading existing index from %s", INDEX_PATH)
        rag.load(str(INDEX_PATH))
    yield
    logger.info("Saving index to %s", INDEX_PATH)
    rag.save(str(INDEX_PATH))

app = FastAPI(title="RAGForge API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 20
    top_n: int = 5

class ChunkResult(BaseModel):
    doc_id: str
    chunk_index: int
    chunk_text: str
    relevance_score: float
    diversity_rank: int

class QueryResponse(BaseModel):
    query: str
    chunks: list[ChunkResult]
    llm_answer: str | None = None
    llm_available: bool

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_indexed: int

class StatusResponse(BaseModel):
    total_chunks: int
    doc_ids: list[str]
    llm_available: bool

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/status", response_model=StatusResponse)
async def status():
    stats = rag.stats()
    return StatusResponse(
        total_chunks=stats["total_chunks"],
        doc_ids=stats["doc_ids"],
        llm_available=bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")),
    )


@app.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md", ".docx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")

    doc_id = str(uuid.uuid4())[:8] + "_" + Path(file.filename).stem
    raw_bytes = await file.read()

    try:
        text = parse_document(raw_bytes, ext)
    except Exception as e:
        raise HTTPException(422, f"Failed to parse document: {e}")

    if not text.strip():
        raise HTTPException(422, "Document produced no extractable text")

    try:
        n_chunks = rag.ingest(text, doc_id)
    except Exception as e:
        raise HTTPException(500, f"Indexing failed: {e}")

    logger.info("Indexed %s → %d chunks (doc_id=%s)", file.filename, n_chunks, doc_id)
    return IngestResponse(doc_id=doc_id, filename=file.filename, chunks_indexed=n_chunks)


@app.post("/query", response_model=QueryResponse)
async def query_docs(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    results = rag.query(req.query, top_k=req.top_k, top_n=req.top_n)

    chunks = [
        ChunkResult(
            doc_id=r["doc_id"],
            chunk_index=r["chunk_index"],
            chunk_text=r["chunk_text"],
            relevance_score=round(r["relevance_score"], 4),
            diversity_rank=int(r["diversity_rank"]),
        )
        for r in results
    ]

    llm_answer = None
    llm_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if llm_key and chunks:
        context = "\n\n---\n\n".join(
            f"[{c.doc_id} chunk {c.chunk_index}]:\n{c.chunk_text}" for c in chunks
        )
        try:
            llm_answer = await stream_answer(req.query, context)
        except Exception as e:
            logger.warning("LLM call failed (falling back to chunks): %s", e)

    return QueryResponse(
        query=req.query,
        chunks=chunks,
        llm_answer=llm_answer,
        llm_available=bool(llm_key),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE streaming endpoint — streams LLM answer token by token if key available."""
    results = rag.query(req.query, top_k=req.top_k, top_n=req.top_n)
    chunks = results[:req.top_n]

    llm_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not llm_key or not chunks:
        # No LLM: stream chunks as SSE events
        async def chunk_stream():
            for r in chunks:
                import json
                yield f"data: {json.dumps({'type': 'chunk', 'data': r})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(chunk_stream(), media_type="text/event-stream")

    context = "\n\n---\n\n".join(
        f"[{r['doc_id']} chunk {r['chunk_index']}]:\n{r['chunk_text']}" for r in chunks
    )

    async def llm_stream():
        import json
        # First send the retrieved chunks
        for r in chunks:
            yield f"data: {json.dumps({'type': 'chunk', 'data': r})}\n\n"
        # Then stream LLM tokens
        async for token in stream_answer(req.query, context, streaming=True):
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(llm_stream(), media_type="text/event-stream")


@app.delete("/index")
async def clear_index():
    global rag
    rag = RAGWrapper()
    return {"status": "cleared"}
