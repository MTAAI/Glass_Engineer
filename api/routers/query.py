"""
Glass Expert AI — Query Router
Full RAG pipeline: retrieve → format context → generate answer with LLM.
"""
import os
import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from loguru import logger
from api.models.schemas import QueryRequest, QueryResponse, SourceChunk

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Ask a glass science question. The system will:
    1. Detect the language (English or Farsi)
    2. Retrieve the most relevant knowledge base chunks
    3. Generate a precise expert answer using the LLM
    4. Return the answer with cited sources
    """
    from retrieval.retriever import retrieve_with_auto_language, format_context
    from retrieval.llm import generate_answer

    start_time = time.time()

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
        # No relevant chunks found — return a graceful no-knowledge response
        return QueryResponse(
            question=request.question,
            answer=(
                "I could not find relevant information in the knowledge base to answer "
                "this question. Please try rephrasing, or this topic may not yet be "
                "covered in the ingested documents."
                if detected_language == "en"
                else
                "اطلاعات مرتبطی در پایگاه دانش برای پاسخ به این سوال یافت نشد. "
                "لطفاً سوال را به شکل دیگری مطرح کنید."
            ),
            sources=[],
            language_detected=detected_language,
            retrieval_time_ms=retrieval_time_ms,
            total_chunks_searched=0,
            model_used="no-retrieval",
        )

    context_block = format_context(chunks)

    # ── Step 3: Generate answer with LLM ──────────────────────────────────────
    try:
        answer, model_used = await generate_answer(
            question=request.question,
            context=context_block,
            language=detected_language,
        )
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        # Fall back to returning just the retrieved context without LLM
        answer = (
            "LLM is not available. Here is the most relevant information from the "
            "knowledge base:\n\n" + context_block
        )
        model_used = "retrieval-only"

    # ── Step 4: Build response ─────────────────────────────────────────────────
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
    )


@router.get("/query/test")
async def test_query():
    """Quick test endpoint — runs a sample query to verify the pipeline."""
    test_request = QueryRequest(
        question="What is the glass transition temperature of borosilicate glass?",
        top_k=3,
    )
    return await query_knowledge_base(test_request)
