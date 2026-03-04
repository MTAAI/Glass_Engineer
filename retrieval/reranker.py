"""
Glass Expert AI — Cross-Encoder Reranker
Uses BAAI/bge-reranker-v2-m3 to re-score retrieved chunks for higher precision.

The retrieval pipeline is:
  1. BAAI/bge-large-en-v1.5 (bi-encoder) → fast approximate retrieval (top_k * 3 candidates)
  2. BAAI/bge-reranker-v2-m3 (cross-encoder) → precise re-scoring → top_k final results

The cross-encoder reads the full query + chunk pair together, giving much more
accurate relevance scores than the bi-encoder cosine similarity alone.
"""
import os
from typing import List, Dict, Any
from functools import lru_cache
from loguru import logger


RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
RERANK_ENABLED  = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_TOP_N    = int(os.getenv("RERANK_TOP_N", "5"))  # Final results after reranking


@lru_cache(maxsize=1)
def _get_reranker():
    """Load and cache the cross-encoder reranker model (singleton)."""
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker model: {RERANKER_MODEL} on {RERANKER_DEVICE}")
        model = CrossEncoder(
            RERANKER_MODEL,
            device=RERANKER_DEVICE,
            max_length=512,
        )
        logger.info("✅ Reranker model loaded successfully.")
        return model
    except ImportError:
        logger.warning("sentence-transformers not installed. Reranking disabled.")
        return None
    except Exception as e:
        logger.warning(f"Failed to load reranker model: {e}. Reranking disabled.")
        return None


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int = None,
) -> List[Dict[str, Any]]:
    """
    Re-rank retrieved chunks using the cross-encoder model.

    Args:
        query:   The user's question
        chunks:  List of chunk dicts from the bi-encoder retriever
        top_n:   Number of top results to return (defaults to RERANK_TOP_N env var)

    Returns:
        Re-ranked list of chunk dicts, sorted by cross-encoder score (descending).
        Each chunk gets a new 'rerank_score' field added.
        Falls back to original order if reranker is unavailable.
    """
    if not RERANK_ENABLED or not chunks:
        return chunks

    if top_n is None:
        top_n = min(RERANK_TOP_N, len(chunks))

    model = _get_reranker()
    if model is None:
        logger.debug("Reranker not available, returning original order.")
        return chunks[:top_n]

    try:
        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, chunk.get("content", "")) for chunk in chunks]

        # Score all pairs — cross-encoder reads both texts together
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        logger.debug(
            f"Reranked {len(chunks)} chunks → top {top_n}. "
            f"Top score: {reranked[0].get('rerank_score', 0):.4f} "
            f"(was similarity: {reranked[0].get('similarity', 0):.4f})"
        )

        return reranked[:top_n]

    except Exception as e:
        logger.warning(f"Reranking failed: {e}. Returning original order.")
        return chunks[:top_n]
