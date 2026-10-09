from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import APIStatusError
from pydantic import BaseModel, Field

from app.agent import ResearchAgent
from app.config import Settings
from app.ingest import index_is_current


logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)


class ChatResponse(BaseModel):
    answer: str
    evidence: list[dict]
    trace: list[dict]


@lru_cache(maxsize=1)
def get_agent() -> ResearchAgent:
    settings = Settings()
    try:
        settings.validate_assets(require_model=True)
        return ResearchAgent(settings)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


app = FastAPI(
    title="Agentic RAG",
    version="2.0.0",
    description="Grounded answers over local documents and optional SQLite data.",
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    settings = Settings()
    try:
        settings.validate_assets()
        assets_ready = True
    except (RuntimeError, ValueError):
        assets_ready = False
    return {
        "status": "ok" if assets_ready else "degraded",
        "assets_ready": assets_ready,
        "index_ready": index_is_current(settings) if assets_ready else False,
        "chat_model_configured": bool(settings.chat_model),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    agent: ResearchAgent = Depends(get_agent),
) -> dict:
    try:
        return agent.answer(request.question).to_dict()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except APIStatusError as error:
        logger.warning("Model provider returned HTTP %s: %s", error.status_code, error.message)
        raise HTTPException(
            status_code=error.status_code,
            detail=f"Model provider error: {error.message}",
        ) from error
    except Exception as error:
        logger.exception("Unable to complete the grounded answer")
        raise HTTPException(
            status_code=502,
            detail=f"Unable to complete the grounded answer: {error}",
        ) from error
