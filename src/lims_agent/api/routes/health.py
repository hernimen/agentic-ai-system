from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma

from lims_agent.core.config import get_settings

router = APIRouter(tags=["health"])


def _check_chroma_ready() -> bool:
    settings = get_settings()
    persist_dir = Path(settings.chroma_persist_dir)
    if not persist_dir.exists():
        return False
    try:
        store = Chroma(
            collection_name=settings.chroma_collection,
            persist_directory=str(persist_dir),
            embedding_function=None,
        )
        store.get(limit=1)
        return True
    except Exception:
        return False


def _check_ollama_ready() -> bool:
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        response.raise_for_status()
        return True
    except Exception:
        return False


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready() -> JSONResponse:
    checks: dict[str, str] = {
        "chroma": "ok",
        "ollama": "ok",
    }

    try:
        if not _check_chroma_ready():
            checks["chroma"] = "error"
    except Exception:
        checks["chroma"] = "error"

    try:
        if not _check_ollama_ready():
            checks["ollama"] = "error"
    except Exception:
        checks["ollama"] = "error"

    is_ready = all(status == "ok" for status in checks.values())
    payload: dict[str, Any] = {"status": "ok" if is_ready else "error", "checks": checks}
    return JSONResponse(status_code=200 if is_ready else 503, content=payload)
