from fastapi import APIRouter
from app.config import LLAMA_SERVER_URL

router = APIRouter()


@router.get("/health")
@router.get("/api/health")
async def health_check():
    """Liveness probe. Returns the configured llama-server URL."""
    return {"status": "ok", "llama_server": LLAMA_SERVER_URL}
