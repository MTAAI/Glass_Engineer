"""
Glass Expert AI — Cross-Encoder Reranker
==========================================
Model: BAAI/bge-reranker-v2-m3

Pipeline position:
  bge-m3 bi-encoder
    → dense ANN search (pgvector HNSW)
    + sparse lexical search (bge-m3 weights → BM25-style)
    → RRF fusion
    → bge-reranker-v2-m3 cross-encoder   ← this file
    → final top-k results

Why bge-reranker-v2-m3:
  - Same family as bge-m3, so query/passage representations are aligned
  - Multilingual: handles Farsi passages natively
  - Significantly more accurate than cosine similarity alone for
    glass science queries with specific terminology
  - Runs on GPU (cuda) if available, gracefully degrades to CPU

Compatible with:
  - dict-style chunks (Engineer Z's branch pattern)
  - RetrievedChunk dataclass (new architecture)
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Union

from loguru import logger


# ── Configuration ──────────────────────────────────────────────────────────────
RERANKER_MODEL  = os.getenv("RERANKER_MODEL",  "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda")
RERANK_ENABLED  = os.getenv("RERANK_ENABLED",  "true").lower() == "true"
RERANK_TOP_N    = int(os.getenv("RERANK_TOP_N", "5"))


# ── Model singleton ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_reranker():
    """Load and cache the cross-encoder — one instance per process."""
    if not RERANK_ENABLED:
        return None

    try:
        from sentence_transformers import CrossEncoder
        logger.info(
            f"Loading reranker: {RERANKER_MODEL} | device={RERANKER_DEVICE}"
        )
        model = CrossEncoder(
            RERANKER_MODEL,
            device=RERANKER_DEVICE,
            max_length=512,
        )
        logger.info("bge-reranker-v2-m3 loaded successfully")
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — reranking disabled. "
            "Install with: pip install sentence-transformers"
        )
        return None
    except Exception as e:
        logger.warning(f"Reranker load failed: {e} — reranking disabled")
        return None


# ── Helpers for dual chunk format ──────────────────────────────────────────────
def _get_content(chunk: Any) -> str:
    """Get content from either dict or dataclass chunk."""
    if isinstance(chunk, dict):
        return chunk.get("content", "")
    return getattr(chunk, "content", "")


def _set_rerank_score(chunk: Any, score: float) -> None:
    """Set rerank_score on either dict or dataclass chunk."""
    if isinstance(chunk, dict):
        chunk["rerank_score"] = score
    else:
        chunk.rerank_score = score


def _get_rerank_score(chunk: Any) -> float:
    if isinstance(chunk, dict):
        return chunk.get("rerank_score", 0.0)
    return getattr(chunk, "rerank_score", 0.0)


def _get_similarity(chunk: Any) -> float:
    if isinstance(chunk, dict):
        return chunk.get("similarity", 0.0)
    return getattr(chunk, "similarity", 0.0)


# ── Public API ─────────────────────────────────────────────────────────────────
def rerank(
    query: str,
    chunks: list,
    top_n: int | None = None,
) -> list:
    """
    Re-rank retrieved chunks using bge-reranker-v2-m3.

    Works with both:
      - list[dict]            — Engineer Z's retriever format
      - list[RetrievedChunk]  — new async retriever format

    Mutates rerank_score on each chunk in-place, then returns sorted list.
    Falls back to original order if the reranker is unavailable.

    Args:
        query:  User's question (original, not expanded)
        chunks: Retrieved chunks to re-rank
        top_n:  Final number of results to return (default: RERANK_TOP_N)

    Returns:
        Re-ranked list, truncated to top_n
    """
    if not chunks:
        return chunks

    n     = top_n or min(RERANK_TOP_N, len(chunks))
    model = _get_reranker()

    if model is None:
        logger.debug("Reranker unavailable — returning original order")
        return chunks[:n]

    try:
        pairs  = [(query, _get_content(c)) for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)

        for chunk, score in zip(chunks, scores):
            _set_rerank_score(chunk, float(score))

        reranked = sorted(chunks, key=_get_rerank_score, reverse=True)

        logger.debug(
            f"Reranked {len(chunks)} → top {n} | "
            f"top_score={_get_rerank_score(reranked[0]):.4f} "
            f"(similarity was {_get_similarity(reranked[0]):.4f})"
        )
        return reranked[:n]

    except Exception as e:
        logger.warning(f"Reranking failed: {e} — returning original order")
        return chunks[:n]
