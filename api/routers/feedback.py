"""
Glass Expert AI — Feedback Router
POST /api/v1/feedback        — Submit overall answer feedback (thumbs up/down + rating)
POST /api/v1/feedback/source — Submit per-source relevance feedback
GET  /api/v1/feedback/stats  — Get aggregated feedback statistics
"""
import os
import psycopg2
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models.schemas import (
    FeedbackRequest, SourceFeedbackRequest, FeedbackResponse
)

router = APIRouter()


def _get_db_conn():
    """Get a database connection."""
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def _ensure_feedback_tables(conn):
    """Create feedback tables if they don't exist."""
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          SERIAL PRIMARY KEY,
            question    TEXT NOT NULL,
            answer      TEXT NOT NULL,
            rating      INTEGER CHECK (rating BETWEEN 1 AND 5),
            helpful     BOOLEAN NOT NULL,
            comment     TEXT,
            query_id    TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS source_feedback (
            id           SERIAL PRIMARY KEY,
            question     TEXT NOT NULL,
            source_title TEXT NOT NULL,
            source_type  TEXT,
            relevant     BOOLEAN NOT NULL,
            comment      TEXT,
            created_at   TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a generated answer.

    - `rating`: 1 (poor) to 5 (excellent)
    - `helpful`: true/false
    - `comment`: optional free-text feedback
    """
    try:
        conn = _get_db_conn()
        _ensure_feedback_tables(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback (question, answer, rating, helpful, comment, query_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                request.question,
                request.answer[:2000],  # Truncate to avoid huge DB entries
                request.rating,
                request.helpful,
                request.comment,
                request.query_id,
            )
        )
        feedback_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Feedback recorded: id={feedback_id}, rating={request.rating}, helpful={request.helpful}")
        return FeedbackResponse(
            success=True,
            message="Thank you for your feedback!",
            feedback_id=feedback_id,
        )
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.post("/feedback/source", response_model=FeedbackResponse)
async def submit_source_feedback(request: SourceFeedbackRequest):
    """
    Submit feedback on whether a specific source was relevant to the question.

    Used to improve retrieval quality over time.
    """
    try:
        conn = _get_db_conn()
        _ensure_feedback_tables(conn)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO source_feedback (question, source_title, source_type, relevant, comment)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                request.question,
                request.source_title,
                request.source_type,
                request.relevant,
                request.comment,
            )
        )
        feedback_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Source feedback recorded: id={feedback_id}, source={request.source_title}, relevant={request.relevant}")
        return FeedbackResponse(
            success=True,
            message="Source feedback recorded.",
            feedback_id=feedback_id,
        )
    except Exception as e:
        logger.error(f"Source feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save source feedback: {str(e)}")


@router.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get aggregated feedback statistics.
    Returns overall ratings, helpfulness rate, and top-rated sources.
    """
    try:
        conn = _get_db_conn()
        _ensure_feedback_tables(conn)
        cur = conn.cursor()

        # Overall feedback stats
        cur.execute("""
            SELECT
                COUNT(*)                                    AS total_feedback,
                ROUND(AVG(rating)::numeric, 2)              AS avg_rating,
                SUM(CASE WHEN helpful THEN 1 ELSE 0 END)    AS helpful_count,
                COUNT(*)                                    AS total_count
            FROM feedback;
        """)
        row = cur.fetchone()
        total, avg_rating, helpful_count, total_count = row

        # Source feedback stats
        cur.execute("""
            SELECT
                source_title,
                COUNT(*)                                        AS total_feedback,
                SUM(CASE WHEN relevant THEN 1 ELSE 0 END)      AS relevant_count
            FROM source_feedback
            GROUP BY source_title
            ORDER BY relevant_count DESC
            LIMIT 10;
        """)
        source_stats = [
            {
                "source": r[0],
                "total_feedback": r[1],
                "relevant_count": r[2],
                "relevance_rate": round(r[2] / r[1] * 100, 1) if r[1] > 0 else 0,
            }
            for r in cur.fetchall()
        ]

        cur.close()
        conn.close()

        helpfulness_rate = round(helpful_count / total_count * 100, 1) if total_count > 0 else 0

        return {
            "total_feedback": total,
            "average_rating": float(avg_rating) if avg_rating else 0.0,
            "helpfulness_rate_pct": helpfulness_rate,
            "top_sources_by_relevance": source_stats,
        }
    except Exception as e:
        logger.error(f"Feedback stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")
