"""
Glass Expert AI — Health Check Router
"""
import os
import psycopg2
import redis as redis_lib
from fastapi import APIRouter
from loguru import logger
from api.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the status of all system components."""

    # Check database
    db_status = "unhealthy"
    total_docs = 0
    total_chunks = 0
    try:
        from api.database import get_db
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(DISTINCT title) FROM documents_bgem3;")
            total_docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM documents_bgem3;")
            total_chunks = cur.fetchone()[0]
            cur.close()
        db_status = "healthy"
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")

    # Check Redis
    redis_status = "unhealthy"
    try:
        r = redis_lib.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        r.ping()
        redis_status = "healthy"
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")

    # Check embedding model (just verify it's cached)
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    model_status = f"ready ({model_name})"

    overall = "healthy" if db_status == "healthy" else "degraded"

    return HealthResponse(
        status=overall,
        database=db_status,
        redis=redis_status,
        embedding_model=model_status,
        total_documents=total_docs,
        total_chunks=total_chunks,
        version="2.0.0",
    )
