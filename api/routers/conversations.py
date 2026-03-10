"""
Glass Expert AI — Conversations Router
POST   /api/v1/conversations              — Create new conversation
GET    /api/v1/conversations              — List all conversations (sidebar)
GET    /api/v1/conversations/{session_id} — Get full conversation with messages
PATCH  /api/v1/conversations/{session_id} — Rename conversation
DELETE /api/v1/conversations/{session_id} — Delete conversation + messages
"""
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models.schemas import (
    ConversationSummary,
    ConversationDetail,
    ChatMessage,
    CreateConversationRequest,
    CreateConversationResponse,
    RenameConversationRequest,
)

router = APIRouter()

ANON_USER_ID = "00000000-0000-0000-0000-000000000001"


def _get_db_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


@router.post("/conversations", response_model=CreateConversationResponse)
async def create_conversation(request: CreateConversationRequest = None):
    """Create a new conversation session."""
    title = (request.title if request and request.title else "New Conversation")
    try:
        conn = _get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            INSERT INTO conversations (user_id, title)
            VALUES (%s, %s)
            RETURNING session_id, title
            """,
            (ANON_USER_ID, title),
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        return CreateConversationResponse(
            session_id=str(row["session_id"]),
            title=row["title"],
        )
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations():
    """List all conversations ordered by most recent activity."""
    try:
        conn = _get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT
                c.session_id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(ch.id) AS message_count
            FROM conversations c
            LEFT JOIN chat_history ch ON ch.session_id = c.session_id
            WHERE c.user_id = %s
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """,
            (ANON_USER_ID,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            ConversationSummary(
                session_id=str(r["session_id"]),
                title=r["title"],
                created_at=r["created_at"].isoformat(),
                updated_at=r["updated_at"].isoformat(),
                message_count=r["message_count"],
            )
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{session_id}", response_model=ConversationDetail)
async def get_conversation(session_id: str):
    """Get a conversation with all its messages."""
    try:
        conn = _get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get conversation metadata
        cur.execute(
            "SELECT session_id, title, created_at, updated_at FROM conversations WHERE session_id = %s",
            (session_id,),
        )
        conv = cur.fetchone()
        if not conv:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get messages
        cur.execute(
            """
            SELECT id, role, content, sources, metadata, created_at
            FROM chat_history
            WHERE session_id = %s
            ORDER BY created_at ASC
            """,
            (session_id,),
        )
        msg_rows = cur.fetchall()
        cur.close()
        conn.close()

        messages = [
            ChatMessage(
                id=str(m["id"]),
                role=m["role"],
                content=m["content"],
                sources=m["sources"] if isinstance(m["sources"], list) else json.loads(m["sources"] or "[]"),
                metadata=m["metadata"] if isinstance(m["metadata"], dict) else json.loads(m["metadata"] or "{}"),
                created_at=m["created_at"].isoformat(),
            )
            for m in msg_rows
        ]

        return ConversationDetail(
            session_id=str(conv["session_id"]),
            title=conv["title"],
            created_at=conv["created_at"].isoformat(),
            updated_at=conv["updated_at"].isoformat(),
            messages=messages,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/conversations/{session_id}")
async def rename_conversation(session_id: str, request: RenameConversationRequest):
    """Rename a conversation."""
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversations SET title = %s WHERE session_id = %s",
            (request.title, session_id),
        )
        if cur.rowcount == 0:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Conversation not found")
        conn.commit()
        cur.close()
        conn.close()
        return {"success": True, "title": request.title}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming conversation {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str):
    """Delete a conversation and all its messages."""
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        # Delete messages first (FK), then conversation
        cur.execute("DELETE FROM chat_history WHERE session_id = %s", (session_id,))
        cur.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
        if cur.rowcount == 0:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Conversation not found")
        conn.commit()
        cur.close()
        conn.close()
        return {"success": True, "message": "Conversation deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
