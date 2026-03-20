"""
Glass Expert AI — Query Router
Full RAG pipeline: retrieve → format context → generate answer with LLM.
Auto-saves Q&A to chat_history when session_id is provided.
"""
import os
import time
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from api.models.schemas import QueryRequest, QueryResponse, SourceChunk
from api.auth import require_auth, UserInToken

router = APIRouter()


def _log_analytics(
    user_id: str,
    query_text: str,
    language: str,
    retrieval_latency_ms: float,
    total_latency_ms: float,
    chunks_retrieved: int,
    top_rerank_score: float,
    fallback_used: bool,
    fallback_reason: str,
    model_used: str,
) -> None:
    """Log query analytics to query_analytics table — fire and forget."""
    try:
        from api.database import get_db
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO query_analytics (
                    user_id, query_text, language,
                    retrieval_latency_ms, total_latency_ms,
                    chunks_retrieved, top_rerank_score,
                    fallback_used, fallback_reason, model_used
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    user_id, query_text[:500], language,
                    round(retrieval_latency_ms, 2), round(total_latency_ms, 2),
                    chunks_retrieved, top_rerank_score,
                    fallback_used, fallback_reason, model_used,
                ),
            )
            conn.commit()
            cur.close()
    except Exception as e:
        logger.debug(f"Analytics logging failed (non-critical): {e}")


def _validate_citations(answer: str, num_sources: int) -> str:
    """
    Validate [Source N] citations in the LLM answer.
    Remove citations that reference non-existent sources.
    Also handle Farsi citations [منبع N].
    """
    import re

    def _replace_invalid(match):
        num = int(match.group(1))
        if 1 <= num <= num_sources:
            return match.group(0)
        return ""

    answer = re.sub(r'\[Source\s+(\d+)\]', _replace_invalid, answer)

    def _replace_invalid_fa(match):
        fa_num = match.group(1)
        digit_map = {'۰':'0','۱':'1','۲':'2','۳':'3','۴':'4','۵':'5','۶':'6','۷':'7','۸':'8','۹':'9'}
        en_num = ''.join(digit_map.get(c, c) for c in fa_num)
        try:
            num = int(en_num)
            if 1 <= num <= num_sources:
                return match.group(0)
        except ValueError:
            pass
        return ""

    answer = re.sub(r'\[منبع\s+([۰-۹0-9]+)\]', _replace_invalid_fa, answer)
    return answer.strip()


def _load_user_memory(user_id: str) -> list:
    """Load user memory entries to personalize LLM responses."""
    try:
        from api.database import get_db
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT key, value FROM user_memory
                WHERE user_id = %s ORDER BY updated_at DESC LIMIT 10
            """, [user_id])
            rows = cur.fetchall()
            cur.close()
            return [{"key": r[0], "value": r[1]} for r in rows]
    except Exception as e:
        logger.warning(f"Could not load user memory: {e}")
        return []


def _load_conversation_history(user_id: str, session_id: str, limit: int = 10) -> list:
    """Load recent messages from chat_history for conversation continuity."""
    try:
        from api.database import get_db
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT role, content FROM chat_history
                WHERE user_id = %s AND session_id = %s
                ORDER BY created_at DESC LIMIT %s
            """, [user_id, session_id, limit])
            rows = cur.fetchall()
            cur.close()
            return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as e:
        logger.warning(f"Could not load conversation history: {e}")
        return []


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest, user: UserInToken = Depends(require_auth)):
    """
    Ask a glass science question. The system will:
    1. Detect the language (English or Farsi)
    2. Retrieve the most relevant knowledge base chunks
    3. Generate a precise expert answer using the LLM
    4. Auto-save Q&A to chat_history (if session_id provided)
    5. Return the answer with cited sources
    """
    from retrieval.retriever import retrieve_with_auto_language, format_context
    from retrieval.llm import generate_answer

    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    conversation_history = []
    user_memory = []
    is_authenticated = user.user_id != "anonymous"
    if is_authenticated and request.session_id:
        conversation_history = _load_conversation_history(user.user_id, session_id)
        user_memory = _load_user_memory(user.user_id)

    # ── Step 1: Retrieve ────────────────────────────────────────────────────────
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

    # ── Step 2: No results early return ────────────────────────────────────────
    if not chunks:
        answer = (
            "I could not find relevant information in the knowledge base to answer "
            "this question. Please try rephrasing, or this topic may not yet be "
            "covered in the ingested documents."
            if detected_language == "en"
            else "اطلاعات مرتبطی در پایگاه دانش برای پاسخ به این سوال یافت نشد. "
                 "لطفاً سوال را به شکل دیگری مطرح کنید."
        )
        total_latency_ms = (time.time() - start_time) * 1000
        _log_analytics(
            user_id=user.user_id,
            query_text=request.question,
            language=detected_language,
            retrieval_latency_ms=retrieval_time_ms,
            total_latency_ms=total_latency_ms,
            chunks_retrieved=0,
            top_rerank_score=0.0,
            fallback_used=True,
            fallback_reason="no-retrieval",
            model_used="no-retrieval",
        )
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=[],
            language_detected=detected_language,
            retrieval_time_ms=round(retrieval_time_ms, 2),
            total_chunks_searched=0,
            model_used="no-retrieval",
            session_id=session_id,
        )

    # ── Step 3: Format context ──────────────────────────────────────────────────
    context_block = format_context(chunks)

    # ── Step 4: Generate answer ─────────────────────────────────────────────────
    try:
        answer, model_used = await generate_answer(
            question=request.question,
            context=context_block,
            language=detected_language,
            conversation_history=conversation_history if conversation_history else None,
            user_memory=user_memory if user_memory else None,
        )
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        answer = (
            "LLM is not available. Here is the most relevant information from the "
            "knowledge base:\n\n" + context_block
        )
        model_used = "retrieval-only"

    # ── Step 5: Validate citations ──────────────────────────────────────────────
    answer = _validate_citations(answer, len(chunks))

    # ── Step 6: Build sources ───────────────────────────────────────────────────
    sources = [
        SourceChunk(
            title=chunk.get("title", "Unknown"),
            source_type=chunk.get("source_type", "unknown"),
            language=chunk.get("language", "en"),
            similarity=round(chunk.get("similarity", 0.0), 4),
            content_preview=chunk.get("content", "")[:300] + "...",
            rerank_score=round(chunk["rerank_score"], 4) if "rerank_score" in chunk else None,
        )
        for chunk in chunks
    ]

    # ── Step 7: Log analytics ───────────────────────────────────────────────────
    total_latency_ms = (time.time() - start_time) * 1000
    top_rerank_score = sources[0].rerank_score if sources and sources[0].rerank_score else 0.0
    fallback_used = model_used in ("retrieval-only", "no-retrieval") or "fallback" in model_used
    _log_analytics(
        user_id=user.user_id,
        query_text=request.question,
        language=detected_language,
        retrieval_latency_ms=retrieval_time_ms,
        total_latency_ms=total_latency_ms,
        chunks_retrieved=len(chunks),
        top_rerank_score=top_rerank_score,
        fallback_used=fallback_used,
        fallback_reason=model_used if fallback_used else "",
        model_used=model_used,
    )

    # ── Step 8: Save to chat_history ────────────────────────────────────────────
    chat_id = None
    if is_authenticated:
        try:
            from api.routers.conversations import save_message
            source_data = [s.model_dump() for s in sources]
            save_message(
                user_id=user.user_id,
                session_id=session_id,
                role="user",
                content=request.question,
            )
            chat_id = save_message(
                user_id=user.user_id,
                session_id=session_id,
                role="assistant",
                content=answer,
                sources=source_data,
                metadata={
                    "model_used": model_used,
                    "language": detected_language,
                    "retrieval_time_ms": round(retrieval_time_ms, 2),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to save chat history: {e}")

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        language_detected=detected_language,
        retrieval_time_ms=round(retrieval_time_ms, 2),
        total_chunks_searched=len(chunks),
        model_used=model_used,
        session_id=session_id,
        chat_id=chat_id,
    )


@router.get("/query/test")
async def test_query():
    """Quick test endpoint — runs a sample query to verify the pipeline."""
    test_request = QueryRequest(
        question="What is the glass transition temperature of borosilicate glass?",
        top_k=3,
    )
    return await query_knowledge_base(test_request)