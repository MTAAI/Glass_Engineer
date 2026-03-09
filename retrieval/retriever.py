"""
Glass Expert AI — Retrieval Module
Implements hybrid search (dense + sparse) with optional re-ranking.
Supports bilingual retrieval (English + Farsi).
"""

import os
import sys
import json
import redis
import hashlib
import psycopg2
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.embedder import embed_query

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
TOP_K                  = int(os.getenv("RETRIEVAL_TOP_K", 20))
FINAL_TOP_K            = int(os.getenv("RETRIEVAL_FINAL_TOP_K", 5))
_SIM_THRESHOLD_DEFAULT = 0.45
REDIS_URL              = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_TTL              = int(os.getenv("REDIS_TTL_SECONDS", 86400))


# ── Database connection ────────────────────────────────────────────────────────
def _get_db_connection():
    from pgvector.psycopg2 import register_vector
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    register_vector(conn)
    return conn


# ── Redis cache connection ─────────────────────────────────────────────────────
def _get_redis():
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        return r
    except Exception:
        logger.warning("Redis not available — caching disabled.")
        return None


# ── Language detection ─────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Detect query language, returning 'en' or 'fa'."""
    try:
        lang = detect(text)
        return "fa" if lang in ("fa", "ar") else "en"
    except LangDetectException:
        return "en"


# ── Cache helpers ──────────────────────────────────────────────────────────────
def _cache_key(query: str, top_k: int, language_filter: str) -> str:
    raw = f"{query}|{top_k}|{language_filter}"
    return f"glass_ai:retrieval:{hashlib.md5(raw.encode()).hexdigest()}"


def _get_cached(query: str, top_k: int, language_filter: str):
    r = _get_redis()
    if not r:
        return None
    key = _cache_key(query, top_k, language_filter)
    cached = r.get(key)
    if cached:
        logger.debug("Cache hit for query")
        return json.loads(cached)
    return None


def _set_cache(query: str, top_k: int, language_filter: str, results: list):
    r = _get_redis()
    if not r:
        return
    key = _cache_key(query, top_k, language_filter)
    r.setex(key, REDIS_TTL, json.dumps(results))


# ── Core retrieval ─────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_k: int = None,
    language_filter: str = None,
    source_type_filter: str = None,
    use_cache: bool = True,
) -> list:
    """
    Retrieve the most relevant document chunks for a query using dense search.
    """
    top_k = top_k or FINAL_TOP_K

    # Check cache first
    if use_cache:
        cached = _get_cached(query, top_k, str(language_filter))
        if cached:
            return cached

    # Generate query embedding
    logger.debug(f"Generating embedding for query: '{query[:80]}...'")
    query_emb = embed_query(query)
    dense_vec = query_emb["dense"]

    # Build SQL query with optional filters
    filters = []
    params = []

    if language_filter:
        filters.append("language = %s")
        params.append(language_filter)

    if source_type_filter:
        filters.append("source_type = %s")
        params.append(source_type_filter)

    where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

    sql = f"""
        SELECT
            id::text,
            title,
            source_type,
            language,
            content,
            metadata,
            1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    params_final = [dense_vec.tolist()] + params + [dense_vec.tolist(), top_k * 4]

    conn = _get_db_connection()
    cur = conn.cursor()
    rows = []
    try:
        cur.execute(sql, params_final)
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    # Build result list and filter by similarity threshold
    sim_threshold = float(os.getenv("SIMILARITY_THRESHOLD", str(_SIM_THRESHOLD_DEFAULT)))
    results = []
    for row in rows:
        similarity = float(row[6])
        if similarity < sim_threshold:
            continue
        results.append({
            "id":          row[0],
            "title":       row[1],
            "source_type": row[2],
            "language":    row[3],
            "content":     row[4],
            "metadata":    row[5] if isinstance(row[5], dict) else {},
            "similarity":  round(similarity, 4),
        })

    # ── Reranking (cross-encoder) ────────────────────────────────────────────
    try:
        from retrieval.reranker import rerank, RERANK_ENABLED
        if RERANK_ENABLED and len(results) > 1:
            results = rerank(query, results, top_n=top_k)
            logger.debug(f"Reranking applied: {len(results)} chunks after rerank")
        else:
            results = results[:top_k]
    except Exception as e:
        logger.warning(f"Reranking skipped: {e}")
        results = results[:top_k]

    # Cache the results
    if use_cache and results:
        _set_cache(query, top_k, str(language_filter), results)

    logger.info(f"Retrieved {len(results)} chunks for query (threshold={sim_threshold})")
    return results


def retrieve_with_auto_language(
    query: str,
    top_k: int = None,
    source_type_filter: str = None,
    language_override: str = None,
) -> tuple:
    """
    Retrieve results with automatic language detection.
    Returns (results, detected_language).
    """
    if language_override and language_override != "auto":
        language = language_override
    else:
        language = detect_language(query)
    logger.info(f"Detected query language: {language.upper()}")

    results = retrieve(
        query,
        top_k=top_k,
        language_filter=None,
        source_type_filter=source_type_filter,
    )
    return results, language


def format_context_for_llm(results: list) -> str:
    """
    Format retrieved chunks into a structured context block for the LLM prompt.
    """
    if not results:
        return "No relevant information found in the knowledge base."

    context_parts = ["KNOWLEDGE BASE CONTEXT:", "=" * 50]

    for i, result in enumerate(results, 1):
        context_parts.append(
            f"\n[Source {i}] {result['title']} "
            f"(Type: {result['source_type']} | "
            f"Language: {result['language'].upper()} | "
            f"Relevance: {result['similarity']:.0%})"
        )
        context_parts.append(result["content"])
        context_parts.append("-" * 40)

    return "\n".join(context_parts)


# Alias used by query router
format_context = format_context_for_llm
