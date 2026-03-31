"""FastAPI entrypoint for the Confident Group RAG chatbot."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import EMBEDDING_MODEL
from app.rag_pipeline import RAGPipeline


rag_pipeline: RAGPipeline | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question for the chatbot")


class ChatResponse(BaseModel):
    response: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    global rag_pipeline
    rag_pipeline = RAGPipeline(embedding_model=EMBEDDING_MODEL)
    yield


app = FastAPI(
    title="Confident Group RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized.")

    try:
        response = rag_pipeline.ask(request.question)
        return ChatResponse(response=response)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime protection
        raise HTTPException(status_code=500, detail=f"Chat request failed: {exc}") from exc
