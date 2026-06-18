"""
RAGForge Backend — FastAPI
"""

import os
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from doc_parser import parse_document
from rag_wrapper import RAGWrapper
from llm_client import stream_answer
from auth import require_auth, create_login_token, LoginRequest, LoginResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragforge")

rag: RAGWrapper | None = None
INDEX_PATH = Path(os.getenv("INDEX_PATH", "./index/ragforge"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    rag = RAGWrapper()
    meta_file = INDEX_PATH.parent / (INDEX_PATH.name + ".meta.json")
    py_file   = INDEX_PATH.parent / (INDEX_PATH.name + ".py.json")
    if meta_file.exists() or py_file.exists():
        try:
            logger.info("Loading existing index from %s", INDEX_PATH)
            rag.load(str(INDEX_PATH))
        except Exception as e:
            logger.warning("Failed to load index (starting fresh): %s", e)
            rag = RAGWrapper()
    yield
    try:
        rag.save(str(INDEX_PATH))
    except Exception as e:
        logger.warning("Failed to save index: %s", e)

app = FastAPI(title="RAGForge API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth endpoint (public) ────────────────────────────────────────────────────
@app.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    return create_login_token(req.password)

# ── Protected models ──────────────────────────────────────────────────────────
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

# ── Protected endpoints ───────────────────────────────────────────────────────
@app.get("/status", response_model=StatusResponse, dependencies=[Depends(require_auth)])
async def status():
    stats = rag.stats()
    return StatusResponse(
        total_chunks=stats["total_chunks"],
        doc_ids=stats["doc_ids"],
        llm_available=bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")),
    )

@app.post("/upload", response_model=IngestResponse, dependencies=[Depends(require_auth)])
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md", ".docx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    doc_id   = str(uuid.uuid4())[:8] + "_" + Path(file.filename).stem
    raw      = await file.read()
    try:
        text = parse_document(raw, ext)
    except Exception as e:
        raise HTTPException(422, f"Failed to parse: {e}")
    if not text.strip():
        raise HTTPException(422, "Document produced no text")
    n = rag.ingest(text, doc_id)
    logger.info("Indexed %s → %d chunks", file.filename, n)
    return IngestResponse(doc_id=doc_id, filename=file.filename, chunks_indexed=n)

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_auth)])
async def query_docs(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    results = rag.query(req.query, top_k=req.top_k, top_n=req.top_n)
    chunks  = [
        ChunkResult(
            doc_id=r["doc_id"], chunk_index=r["chunk_index"],
            chunk_text=r["chunk_text"],
            relevance_score=round(r["relevance_score"], 4),
            diversity_rank=int(r["diversity_rank"]),
        ) for r in results
    ]
    llm_answer = None
    llm_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    if llm_key and chunks:
        context = "\n\n---\n\n".join(
            f"[{c.doc_id} chunk {c.chunk_index}]:\n{c.chunk_text}" for c in chunks)
        try:
            llm_answer = await stream_answer(req.query, context)
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
    return QueryResponse(
        query=req.query, chunks=chunks,
        llm_answer=llm_answer,
        llm_available=bool(llm_key),
    )

@app.delete("/index", dependencies=[Depends(require_auth)])
async def clear_index():
    rag.clear()
    return {"status": "cleared"}
