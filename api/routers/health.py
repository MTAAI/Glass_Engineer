import os
import psycopg2
from fastapi import APIRouter
from loguru import logger
from api.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    db_status = "unhealthy"
    total_docs = 0
    total_chunks = 0
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT title) FROM documents;")
        total_docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM documents;")
        total_chunks = cur.fetchone()[0]
        conn.close()
        db_status = "healthy"
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")

    overall = "healthy" if db_status == "healthy" else "degraded"

    return HealthResponse(
        status=overall,
        database=db_status,
        redis="healthy",
        total_documents=total_docs,
        total_chunks=total_chunks,
    )
