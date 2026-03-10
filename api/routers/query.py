"""
Glass Expert AI — Query Router
Full RAG pipeline: retrieve → format context → generate answer with LLM.
Supports conversation persistence via session_id.
"""
import os
import json
import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from loguru import logger
import psycopg2
from psycopg2.extras import RealDictCursor
from api.models.schemas import QueryRequest, QueryResponse, SourceChunk

router = APIRouter()

ANON_USER_ID = "00000000-0000-0000-0000-000000000001"


def _get_db_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def _load_conversation_history(session_id: str) -> list[dict]:
    """Load last 10 messages from chat_history for a session."""
    try:
        conn = _get_db_conn()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT role, content FROM chat_history
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (session_id,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        # Reverse to chronological order
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception as e:
        logger.warning(f"Could not load conversation history: {e}")
        return []


def _save_message(session_id: str, role: str, content: str, sources: list = None, metadata: dict = None):
    """Save a message to chat_history."""
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chat_history (user_id, session_id, role, content, sources, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                ANON_USER_ID,
                session_id,
                role,
                content,
                json.dumps(sources or []),
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not save message: {e}")


def _auto_title_conversation(session_id: str, first_message: str):
    """Set conversation title from first user message (truncated to 80 chars)."""
    title = first_message[:80].strip()
    if len(first_message) > 80:
        title += "..."
    try:
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE conversations SET title = %s
            WHERE session_id = %s AND title = 'New Conversation'
            """,
            (title, session_id),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not auto-title conversation: {e}")


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Ask a glass science question. The system will:
    1. Detect the language (English or Farsi)
    2. Retrieve the most relevant knowledge base chunks
    3. Generate a precise expert answer using the LLM
    4. Return the answer with cited sources

    If session_id is provided, loads conversation history for multi-turn context
    and saves both user + assistant messages to chat_history.
    """
    from retrieval.retriever import retrieve_with_auto_language, format_context
    from retrieval.llm import generate_answer

    start_time = time.time()
    session_id = request.session_id

    # ── Load conversation history if session_id provided ──────────────────────
    conversation_history = []
    if session_id:
        conversation_history = _load_conversation_history(session_id)

    # ── Step 1: Retrieve relevant chunks ──────────────────────────────────────
    try:
        chunks, detected_language = retrieve_with_auto_language(
            query=request.question,
            top_k=request.top_k,
            source_type_filter=request.source_type,
            language_override=request.language,
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    retrieval_time_ms = (time.time() - start_time) * 1000

    # ── Step 2: Format context for LLM ────────────────────────────────────────
    if not chunks:
        no_result_answer = (
            "I could not find relevant information in the knowledge base to answer "
            "this question. Please try rephrasing, or this topic may not yet be "
            "covered in the ingested documents."
            if detected_language == "en"
            else
            "اطلاعات مرتبطی در پایگاه دانش برای پاسخ به این سوال یافت نشد. "
            "لطفاً سوال را به شکل دیگری مطرح کنید."
        )
        # Save messages even for no-result queries
        if session_id:
            _save_message(session_id, "user", request.question)
            _save_message(session_id, "assistant", no_result_answer)
            _auto_title_conversation(session_id, request.question)
        return QueryResponse(
            question=request.question,
            answer=no_result_answer,
            sources=[],
            language_detected=detected_language,
            retrieval_time_ms=retrieval_time_ms,
            total_chunks_searched=0,
            model_used="no-retrieval",
            session_id=session_id,
        )

    context_block = format_context(chunks)

    # ── Step 3: Generate answer with LLM ──────────────────────────────────────
    try:
        answer, model_used = await generate_answer(
            question=request.question,
            context=context_block,
            language=detected_language,
            conversation_history=conversation_history if conversation_history else None,
        )
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        answer = (
            "LLM is not available. Here is the most relevant information from the "
            "knowledge base:\n\n" + context_block
        )
        model_used = "retrieval-only"

    # ── Step 4: Save messages to chat_history ─────────────────────────────────
    if session_id:
        _save_message(session_id, "user", request.question)
        source_data = [
            {"title": c.get("title"), "source_type": c.get("source_type"), "similarity": c.get("similarity")}
            for c in chunks
        ]
        _save_message(
            session_id, "assistant", answer,
            sources=source_data,
            metadata={"model_used": model_used, "retrieval_time_ms": round(retrieval_time_ms, 2)},
        )
        _auto_title_conversation(session_id, request.question)

    # ── Step 5: Build response ─────────────────────────────────────────────────
    sources = [
        SourceChunk(
            title=chunk.get("title", "Unknown"),
            source_type=chunk.get("source_type", "unknown"),
            language=chunk.get("language", "en"),
            similarity=round(chunk.get("similarity", 0.0), 4),
            content_preview=chunk.get("content", "")[:300] + "...",
        )
        for chunk in chunks
    ]

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        language_detected=detected_language,
        retrieval_time_ms=round(retrieval_time_ms, 2),
        total_chunks_searched=len(chunks),
        model_used=model_used,
        session_id=session_id,
    )


@router.get("/query/test")
async def test_query():
    """Quick test endpoint — runs a sample query to verify the pipeline."""
    test_request = QueryRequest(
        question="What is the glass transition temperature of borosilicate glass?",
        top_k=3,
    )
    return await query_knowledge_base(test_request)
