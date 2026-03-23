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
from api.database import get_db_conn, return_db_conn

router = APIRouter()

psycopg2.extras.register_uuid()


def _get_db():
    return get_db_conn()


def _return_db(conn):
    return_db_conn(conn)


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
                MIN(created_at)::text AS first_message_at,
                (SELECT metadata->>'custom_title'
                 FROM chat_history h2
                 WHERE h2.session_id = chat_history.session_id
                   AND h2.metadata->>'custom_title' IS NOT NULL
                 LIMIT 1) AS custom_title
            FROM chat_history
            WHERE user_id = %s
            GROUP BY session_id
            ORDER BY MAX(created_at) DESC
            LIMIT 50
        """, [user.user_id])
        rows = cur.fetchall()
    finally:
        cur.close()
        _return_db(conn)

    conversations = []
    for row in rows:
        title = (row[5] or row[1] or "Untitled conversation")[:80]
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
        _return_db(conn)

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


@router.patch("/conversations/{session_id}")
async def rename_conversation(
    session_id: str,
    body: dict,
    user: UserInToken = Depends(require_auth),
):
    """Rename a conversation by updating the metadata of its first message."""
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")

    conn = _get_db()
    cur = conn.cursor()
    try:
        # Update metadata of the first message in this session to include custom title
        cur.execute("""
            UPDATE chat_history
            SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('custom_title', %s)
            WHERE id = (
                SELECT id FROM chat_history
                WHERE user_id = %s AND session_id = %s
                ORDER BY created_at ASC
                LIMIT 1
            )
        """, [title, user.user_id, session_id])
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conn.commit()
    finally:
        cur.close()
        _return_db(conn)

    return {"success": True, "title": title}


@router.get("/conversations/search")
async def search_conversations(
    q: str,
    limit: int = 20,
    user: UserInToken = Depends(require_auth),
):
    """Search across all user's conversations by keyword. Returns matching messages with context."""
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Search query must be at least 2 characters")

    conn = _get_db()
    cur = conn.cursor()
    try:
        # Search chat_history content using ILIKE for simplicity
        # Returns matching messages grouped by conversation
        search_pattern = f"%{q.strip()}%"
        cur.execute("""
            SELECT
                ch.session_id::text,
                ch.role,
                ch.content,
                ch.created_at::text,
                (SELECT MIN(c2.content) FROM chat_history c2
                 WHERE c2.session_id = ch.session_id AND c2.role = 'user'
                 AND c2.user_id = %s) AS conversation_title,
                (SELECT metadata->>'custom_title' FROM chat_history c3
                 WHERE c3.session_id = ch.session_id
                   AND c3.metadata->>'custom_title' IS NOT NULL
                   AND c3.user_id = %s
                 LIMIT 1) AS custom_title
            FROM chat_history ch
            WHERE ch.user_id = %s
              AND ch.content ILIKE %s
            ORDER BY ch.created_at DESC
            LIMIT %s
        """, [user.user_id, user.user_id, user.user_id, search_pattern, limit])
        rows = cur.fetchall()
    finally:
        cur.close()
        _return_db(conn)

    results = []
    for row in rows:
        content = row[2] or ""
        # Create snippet around the match
        lower_content = content.lower()
        match_pos = lower_content.find(q.strip().lower())
        if match_pos >= 0:
            start = max(0, match_pos - 60)
            end = min(len(content), match_pos + len(q) + 60)
            snippet = ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")
        else:
            snippet = content[:150] + ("..." if len(content) > 150 else "")

        results.append({
            "session_id": row[0],
            "role": row[1],
            "snippet": snippet,
            "created_at": row[3],
            "title": (row[5] or row[4] or "Untitled")[:80],
        })

    return {"query": q, "results": results, "total": len(results)}


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
        _return_db(conn)

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
) -> str | None:
    """Save a single message to chat_history. Returns the chat_history UUID."""
    import json
    conn = _get_db()
    cur = conn.cursor()
    chat_id = None
    try:
        cur.execute("""
            INSERT INTO chat_history (user_id, session_id, role, content, sources, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
            RETURNING id
        """, [
            user_id,
            session_id,
            role,
            content,
            json.dumps(sources or []),
            json.dumps(metadata or {}),
        ])
        row = cur.fetchone()
        chat_id = str(row[0]) if row else None
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to save chat message: {e}")
        conn.rollback()
    finally:
        cur.close()
        _return_db(conn)
    return chat_id


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
        _return_db(conn)

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
        _return_db(conn)

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
        _return_db(conn)

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Memory entry not found")

    return {"success": True, "deleted": key}
