"""FastAPI application — stateless /chat and instant /health."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.conversation.engine import chat_with_timeout
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retrieval import get_retriever

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _warm_retriever_background() -> None:
    try:
        get_retriever().warm()
        log.info("Retriever warmed in background")
    except Exception as exc:
        log.warning("Background retriever warm failed (will retry on /chat): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # /health returns immediately; warm FAISS index in a daemon thread so first /chat
    # is fast without blocking the health check (eval allows ~2 min process start).
    log.info("Service started (/health ready; warming retriever in background)")
    threading.Thread(target=_warm_retriever_background, daemon=True).start()
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return chat_with_timeout(request)
