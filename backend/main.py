"""FastAPI app: startup loading, the two API endpoints, and the test page.

Run from the repo root with: uvicorn backend.main:app
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.confidence import score_query
from backend.generation import Generator
from backend.retrieval import Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backend.main")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Filled in at startup; kept in a dict so the endpoints can reach them
# without module-level model loading at import time.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both models and the corpus once, before serving any request.
    retriever = Retriever()
    logger.info(
        "Corpus live: %s (%d entries)",
        retriever.corpus_path,
        len(retriever.entries),
    )
    state["retriever"] = retriever
    state["generator"] = Generator()
    logger.info("Startup complete, ready to serve")
    yield
    state.clear()


app = FastAPI(title="Streaming RAG backend", lifespan=lifespan)


class RetrieveRequest(BaseModel):
    text: str
    k: int = Field(default=5, ge=1, le=50)


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1)
    doc_ids: list[str] = Field(min_length=1)


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    if not req.text.strip():
        return {
            "query": req.text,
            "results": [],
            "top1_score": 0.0,
            "top2_score": 0.0,
            "confidence": 0.0,
            "decision": "WAIT",
        }

    retriever: Retriever = state["retriever"]
    try:
        results = retriever.search(req.text, req.k)
    except Exception:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail="Retrieval failed")

    top1 = results[0]["score"] if len(results) > 0 else 0.0
    top2 = results[1]["score"] if len(results) > 1 else 0.0
    confidence, decision = score_query(top1, top2)

    return {
        "query": req.text,
        "results": results,
        "top1_score": top1,
        "top2_score": top2,
        "confidence": confidence,
        "decision": decision,
    }


@app.post("/answer")
def answer(req: AnswerRequest) -> dict:
    retriever: Retriever = state["retriever"]
    entries = retriever.get_entries(req.doc_ids)
    if not entries:
        raise HTTPException(status_code=400, detail="No known doc_ids provided")

    generator: Generator = state["generator"]
    try:
        answer_text = generator.answer(req.query, entries)
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail="Answer generation failed")

    return {
        "answer": answer_text,
        "sources": [
            {"doc_id": entry["id"], "title": entry["title"]} for entry in entries
        ],
    }


# Mounted last so the API routes above are matched first.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
