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
    """Log query analytics to query_analytics table — fire and forget.
    Non-critical: if table doesn't exist or insert fails, log and move on."""
    try:
        from api.database import get_db
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO query_analytics (
                    user_id, query_text, language,
                    retrieval_latency_ms, total_latency_ms,
                    chunks_retrieved, top_rerank_score,
                    fallback_used, fallback_reason, model_used
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                user_id, query_text[:500], language,
                round(retrieval_latency_ms, 2), round(total_latency_ms, 2),
                chunks_retrieved, top_rerank_score,
                fallback_used, fallback_reason, model_used,
            ))
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
            return match.group(0)  # valid — keep it
        return ""  # invalid — remove

    # English: [Source 1], [Source 2], etc.
    answer = re.sub(r'\[Source\s+(\d+)\]', _replace_invalid, answer)
    # Farsi: [منبع ۱], [منبع ۲], etc. (convert Persian digits)
    def _replace_invalid_fa(match):
        fa_num = match.group(1)
        # Convert Persian/Arabic digits to int
        digit_map = {'۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
                     '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9'}
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
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT 10
            """, [user_id])
            rows = cur.fetchall()
            cur.close()
        return [{"key": r[0], "value": r[1]} for r in rows]
    except Exception as e:
        logger.warning(f"Could not load user memory: {e}")
        return []


def _load_conversation_history(user_id: str, session_id: str, limit: int = 20) -> list:
    """
    Load conversation history with smart tiered loading for continuity.

    Strategy:
      - Load up to 20 recent messages (10 Q&A turns)
      - Recent messages (last 6) kept in full
      - Older messages: keep user questions full, summarize assistant answers
      - This gives the LLM enough context to understand the conversation flow
        without blowing the token budget
    """
    try:
        from api.database import get_db
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT role, content, created_at FROM chat_history
                WHERE user_id = %s AND session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, [user_id, session_id, limit])
            rows = cur.fetchall()
            cur.close()

        if not rows:
            return []

        # Reverse to chronological order
        messages = [{"role": r[0], "content": r[1]} for r in reversed(rows)]

        # Smart compression: keep recent full, compress older
        if len(messages) > 6:
            compressed = []
            cutoff = len(messages) - 6  # last 6 messages kept full

            for i, msg in enumerate(messages):
                if i < cutoff:
                    content = msg["content"]
                    # Strip old RAG context from user messages
                    if "Reference information:" in content:
                        # Extract just the question
                        parts = content.split("Based on the reference information above, answer this question:")
                        content = parts[-1].strip() if len(parts) > 1 else content[:200]
                    # Summarize older assistant answers to key points
                    if msg["role"] == "assistant" and len(content) > 300:
                        # Keep first 250 chars (usually the direct answer) + last sentence
                        first_part = content[:250].rsplit(". ", 1)[0] + "."
                        content = first_part + " [...]"
                    compressed.append({"role": msg["role"], "content": content})
                else:
                    compressed.append(msg)
            return compressed

        return messages
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

    # Auto-create session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # Load conversation history + user memory for continuity
    conversation_history = []
    user_memory = []
    is_authenticated = user.user_id != "anonymous"
    if is_authenticated:
        if request.session_id:
            conversation_history = _load_conversation_history(user.user_id, session_id)
        user_memory = _load_user_memory(user.user_id)

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
        answer = (
            "I could not find relevant information in the knowledge base to answer "
            "this question. Please try rephrasing, or this topic may not yet be "
            "covered in the ingested documents."
            if detected_language == "en"
            else
            "اطلاعات مرتبطی در پایگاه دانش برای پاسخ به این سوال یافت نشد. "
            "لطفاً سوال را به شکل دیگری مطرح کنید."
        )
        return QueryResponse(
            question=request.question,
            answer=answer,
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
            user_memory=user_memory if user_memory else None,
        )
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        answer = (
            "LLM is not available. Here is the most relevant information from the "
            "knowledge base:\n\n" + context_block
        )
        model_used = "retrieval-only"

    # ── Step 3.5: Validate citations ─────────────────────────────────────────
    answer = _validate_citations(answer, len(chunks))

    # ── Step 4: Build response ─────────────────────────────────────────────────
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

    # ── Step 5: Auto-save to chat_history ─────────────────────────────────────
    chat_id = None
    if is_authenticated:
        try:
            from api.routers.conversations import save_message
            source_data = [s.model_dump() for s in sources]
            # Save user question
            save_message(
                user_id=user.user_id,
                session_id=session_id,
                role="user",
                content=request.question,
            )
            # Save assistant answer — capture chat_id for feedback
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

    total_time_ms = (time.time() - start_time) * 1000

    # ── Step 6: Log query analytics (fire and forget) ─────────────────────
    top_rerank = max((c.get("rerank_score", 0) for c in chunks), default=0)
    is_fallback = "(fallback)" in model_used
    _log_analytics(
        user_id=user.user_id,
        query_text=request.question,
        language=detected_language,
        retrieval_latency_ms=retrieval_time_ms,
        total_latency_ms=total_time_ms,
        chunks_retrieved=len(chunks),
        top_rerank_score=round(top_rerank, 4),
        fallback_used=is_fallback,
        fallback_reason="local_unavailable" if is_fallback else "",
        model_used=model_used,
    )

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
