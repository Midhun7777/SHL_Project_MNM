"""FastAPI application — stateless /chat and instant /health."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.conversation.engine import chat_with_timeout
from app.models import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # No eager model load — keeps Render 512MB free tier stable at startup.
    # FAISS index + fastembed load lazily on first /chat only.
    log.info("Service started (/health ready; retriever loads on first /chat)")
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return chat_with_timeout(request)
