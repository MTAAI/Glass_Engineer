"""
Glass Expert AI — Conversations Router
CRUD for chat sessions + user memory.
"""
import os
import uuid
import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from api.models.schemas import (
    ConversationListResponse, ConversationSummary,
    ConversationDetail, ChatMessage,
    ConversationCreateRequest, ConversationCreateResponse,
    UserMemoryListResponse, UserMemoryEntry,
    UserMemorySaveRequest,
)
from api.auth import require_auth, UserInToken

router = APIRouter()

psycopg2.extras.register_uuid()


def _get_db():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


# ── Conversations ─────────────────────────────────────────────────────────────

@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(user: UserInToken = Depends(require_auth)):
    """List all conversations for the authenticated user, newest first."""
    conn = _get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                session_id::text,
                MIN(content) FILTER (WHERE role = 'user') AS first_question,
                COUNT(*) AS message_count,
                MAX(created_at)::text AS last_message_at,
                MIN(created_at)::text AS first_message_at
            FROM chat_history
            WHERE user_id = %s
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC
            LIMIT 50
        """, [user.user_id])
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    conversations = []
    for row in rows:
        title = (row[1] or "Untitled conversation")[:80]
        conversations.append(ConversationSummary(
            session_id=row[0],
            title=title,
            message_count=row[2],
            last_message_at=row[3],
        ))

    return ConversationListResponse(conversations=conversations)


@router.post("/conversations", response_model=ConversationCreateResponse)
async def create_conversation(
    request: ConversationCreateRequest = None,
    user: UserInToken = Depends(require_auth),
):
    """Create a new conversation session. Returns the session_id."""
    session_id = str(uuid.uuid4())
    title = (request.title if request and request.title else "New conversation")
    return ConversationCreateResponse(session_id=session_id, title=title)


@router.get("/conversations/{session_id}", response_model=ConversationDetail)
async def get_conversation(session_id: str, user: UserInToken = Depends(require_auth)):
    """Get all messages in a conversation."""
    conn = _get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                id::text, role, content, sources, metadata, created_at::text
            FROM chat_history
            WHERE user_id = %s AND session_id = %s
            ORDER BY created_at ASC
        """, [user.user_id, session_id])
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = []
    for row in rows:
        messages.append(ChatMessage(
            id=row[0],
            role=row[1],
            content=row[2],
            sources=row[3] if row[3] else None,
            metadata=row[4] if row[4] else None,
            created_at=row[5],
        ))

    title = next((m.content[:80] for m in messages if m.role == "user"), "Untitled")

    return ConversationDetail(
        session_id=session_id,
        title=title,
        messages=messages,
        created_at=messages[0].created_at,
        last_message_at=messages[-1].created_at,
    )


@router.delete("/conversations/{session_id}")
async def delete_conversation(session_id: str, user: UserInToken = Depends(require_auth)):
    """Delete a conversation and all its messages."""
    conn = _get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM chat_history WHERE user_id = %s AND session_id = %s",
            [user.user_id, session_id],
        )
        deleted = cur.rowcount
        conn.commit()
    finally:
        cur.close()
        conn.close()

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"success": True, "deleted_messages": deleted}


# ── Helper: save message to chat_history ──────────────────────────────────────

def save_message(
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    sources: list = None,
    metadata: dict = None,
):
    """Save a single message to chat_history. Called from query router."""
    import json
    conn = _get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_history (user_id, session_id, role, content, sources, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
        """, [
            user_id,
            session_id,
            role,
            content,
            json.dumps(sources or []),
            json.dumps(metadata or {}),
        ])
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to save chat message: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


# ── User Memory ───────────────────────────────────────────────────────────────

@router.get("/user/memory", response_model=UserMemoryListResponse)
async def get_user_memory(user: UserInToken = Depends(require_auth)):
    """Get all memory entries for the authenticated user."""
    conn = _get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id::text, memory_type, key, value, updated_at::text
            FROM user_memory
            WHERE user_id = %s
            ORDER BY updated_at DESC
        """, [user.user_id])
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    entries = [
        UserMemoryEntry(id=r[0], memory_type=r[1], key=r[2], value=r[3], updated_at=r[4])
        for r in rows
    ]
    return UserMemoryListResponse(entries=entries)


@router.post("/user/memory")
async def save_user_memory(
    request: UserMemorySaveRequest,
    user: UserInToken = Depends(require_auth),
):
    """Save or update a user memory entry (upsert by user_id + key)."""
    conn = _get_db()
    cur = conn.cursor()
    try:
        # Upsert: update if key exists, insert otherwise
        cur.execute("""
            SELECT id FROM user_memory WHERE user_id = %s AND key = %s
        """, [user.user_id, request.key])
        existing = cur.fetchone()

        if existing:
            cur.execute("""
                UPDATE user_memory
                SET value = %s, memory_type = %s, updated_at = NOW()
                WHERE user_id = %s AND key = %s
            """, [request.value, request.memory_type, user.user_id, request.key])
        else:
            cur.execute("""
                INSERT INTO user_memory (user_id, memory_type, key, value)
                VALUES (%s, %s, %s, %s)
            """, [user.user_id, request.memory_type, request.key, request.value])

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save user memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

    return {"success": True, "key": request.key}


@router.delete("/user/memory/{key}")
async def delete_user_memory(key: str, user: UserInToken = Depends(require_auth)):
    """Delete a specific user memory entry by key."""
    conn = _get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM user_memory WHERE user_id = %s AND key = %s",
            [user.user_id, key],
        )
        deleted = cur.rowcount
        conn.commit()
    finally:
        cur.close()
        conn.close()

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Memory entry not found")

    return {"success": True, "deleted": key}
