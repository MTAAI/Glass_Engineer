"""
Glass Expert AI — Feedback Router
POST /api/v1/feedback        — Submit thumbs up/down on a chat message
POST /api/v1/feedback/source — Submit per-source relevance feedback
GET  /api/v1/feedback/stats  — Get aggregated feedback statistics

Schema follows docker/init.sql:
  feedback(id UUID, chat_id UUID FK, user_id UUID FK, rating SMALLINT(-1/1), corrected_text, comment)
  source_feedback(id UUID, feedback_id UUID FK, document_id UUID FK, is_relevant BOOLEAN)
"""
import os
from typing import Optional
from uuid import UUID

import psycopg2
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from api.auth import UserInToken, get_current_user

router = APIRouter()


# ── Schemas (aligned with init.sql) ─────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """Submit thumbs up/down on an assistant message."""
    chat_id: str  # UUID of the chat_history row (assistant message)
    rating: int = Field(..., description="1 = thumbs up, -1 = thumbs down")
    corrected_text: Optional[str] = None
    comment: Optional[str] = None


class SourceFeedbackRequest(BaseModel):
    """Submit per-source relevance feedback (linked to a feedback entry)."""
    feedback_id: str  # UUID of the parent feedback row
    document_id: str  # UUID of the document being rated
    is_relevant: bool


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: str  # UUID string


# ── DB helper ────────────────────────────────────────────────────────────────

def _get_db_conn():
    return psycopg2.connect(os.getenv(
        "DATABASE_URL",
        "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai",
    ))


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    user: UserInToken = Depends(get_current_user),
):
    """
    Submit feedback on an assistant message.
    - `chat_id`: the assistant message UUID from chat_history
    - `rating`:  1 (thumbs up) or -1 (thumbs down)
    - `corrected_text`: optional corrected answer text
    - `comment`: optional free-text comment
    """
    if request.rating not in (-1, 1):
        raise HTTPException(status_code=422, detail="rating must be -1 or 1")

    try:
        conn = _get_db_conn()
        cur = conn.cursor()

        # Verify the chat_id exists
        cur.execute("SELECT id FROM chat_history WHERE id = %s", (request.chat_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Chat message not found")

        cur.execute(
            """
            INSERT INTO feedback (chat_id, user_id, rating, corrected_text, comment)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (request.chat_id, user.user_id, request.rating,
             request.corrected_text, request.comment),
        )
        feedback_id = str(cur.fetchone()[0])
        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"Feedback recorded: id={feedback_id}, chat_id={request.chat_id}, rating={request.rating}")
        return FeedbackResponse(
            success=True,
            message="Thank you for your feedback!",
            feedback_id=feedback_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.post("/feedback/source", response_model=FeedbackResponse)
async def submit_source_feedback(
    request: SourceFeedbackRequest,
    user: UserInToken = Depends(get_current_user),
):
    """
    Submit feedback on whether a specific source document was relevant.
    Linked to an existing feedback entry via feedback_id.
    """
    try:
        conn = _get_db_conn()
        cur = conn.cursor()

        # Verify feedback_id exists
        cur.execute("SELECT id FROM feedback WHERE id = %s", (request.feedback_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Parent feedback not found")

        # Verify document_id exists
        cur.execute("SELECT id FROM documents WHERE id = %s", (request.document_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Document not found")

        cur.execute(
            """
            INSERT INTO source_feedback (feedback_id, document_id, is_relevant)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (request.feedback_id, request.document_id, request.is_relevant),
        )
        sf_id = str(cur.fetchone()[0])
        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"Source feedback recorded: id={sf_id}, doc={request.document_id}, relevant={request.is_relevant}")
        return FeedbackResponse(
            success=True,
            message="Source feedback recorded.",
            feedback_id=sf_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Source feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save source feedback: {str(e)}")


@router.get("/feedback/stats")
async def get_feedback_stats():
    """
    Get aggregated feedback statistics.
    Returns thumbs up/down counts and per-source relevance rates.
    """
    try:
        conn = _get_db_conn()
        cur = conn.cursor()

        # Overall feedback stats
        cur.execute("""
            SELECT
                COUNT(*)                                            AS total,
                SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END)        AS thumbs_up,
                SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END)       AS thumbs_down
            FROM feedback;
        """)
        row = cur.fetchone()
        total, thumbs_up, thumbs_down = row

        # Source relevance stats (top 10 documents by feedback count)
        cur.execute("""
            SELECT
                d.title,
                d.source_type,
                COUNT(*)                                            AS total_votes,
                SUM(CASE WHEN sf.is_relevant THEN 1 ELSE 0 END)    AS relevant_count
            FROM source_feedback sf
            JOIN documents d ON d.id = sf.document_id
            GROUP BY d.id, d.title, d.source_type
            ORDER BY total_votes DESC
            LIMIT 10;
        """)
        source_stats = [
            {
                "title": r[0],
                "source_type": r[1],
                "total_votes": r[2],
                "relevant_count": r[3],
                "relevance_pct": round(r[3] / r[2] * 100, 1) if r[2] > 0 else 0,
            }
            for r in cur.fetchall()
        ]

        cur.close()
        conn.close()

        approval_pct = round(thumbs_up / total * 100, 1) if total and total > 0 else 0

        return {
            "total_feedback": total or 0,
            "thumbs_up": thumbs_up or 0,
            "thumbs_down": thumbs_down or 0,
            "approval_rate_pct": approval_pct,
            "top_sources_by_feedback": source_stats,
        }
    except Exception as e:
        logger.error(f"Feedback stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")
