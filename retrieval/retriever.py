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
TOP_K                  = int(os.getenv("RETRIEVAL_TOP_K", 30))
FINAL_TOP_K            = int(os.getenv("RETRIEVAL_FINAL_TOP_K", 8))
_SIM_THRESHOLD_DEFAULT = 0.35
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


# ── Query translation (for non-English → English retrieval) ──────────────────
_GLOSSARY_CACHE = None

def _load_glossary() -> dict:
    """Load Persian → English glass terminology glossary."""
    global _GLOSSARY_CACHE
    if _GLOSSARY_CACHE is not None:
        return _GLOSSARY_CACHE
    glossary_path = Path(__file__).parent / "glossary_fa_en.json"
    if glossary_path.exists():
        with open(glossary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _GLOSSARY_CACHE = data.get("terms", {})
        logger.info(f"Loaded {len(_GLOSSARY_CACHE)} glossary terms")
    else:
        _GLOSSARY_CACHE = {}
    return _GLOSSARY_CACHE


def _translate_query_to_english(query: str) -> str:
    """Translate a non-English query to English for embedding/retrieval.
    Uses glossary for domain-specific terms + OpenAI for full translation.
    Returns original query if translation fails."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("No OPENAI_API_KEY — skipping query translation")
            return query

        # Build glossary hint from matching terms
        glossary = _load_glossary()
        hints = []
        for fa_term, en_term in glossary.items():
            if fa_term in query:
                hints.append(f"{fa_term} = {en_term}")
        glossary_block = ""
        if hints:
            glossary_block = "\n\nUse these domain-specific translations:\n" + "\n".join(hints)

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Translate the following glass science query to English. Return ONLY the English translation, nothing else.{glossary_block}"},
                {"role": "user", "content": query},
            ],
            max_tokens=200,
            temperature=0,
        )
        translated = resp.choices[0].message.content.strip()
        logger.info(f"Translated query: '{query[:40]}...' → '{translated[:80]}'")
        return translated
    except Exception as e:
        logger.warning(f"Query translation failed: {e}")
        return query


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


# ── Source-type diversity ──────────────────────────────────────────────────────
# Priority source types: these contain general knowledge, standard values,
# textbook definitions — more useful for broad questions.
_PRIORITY_TYPES = {"textbook", "manual", "paper", "sop"}


def _diversify_sources(results: list, top_k: int) -> list:
    """
    Ensure results include the best chunks from EVERY source type, not just
    qa_pair (which outnumbers other types ~5:1 and dominates naive similarity).

    Strategy: Round-robin pick the top chunk from each source type, then fill
    remaining slots by global similarity. This guarantees that textbook
    definitions, manual procedures, paper findings, and SOP standards all
    appear alongside composition-specific simulation data.
    """
    if not results or top_k <= 2:
        return results

    # Group by source_type, preserving similarity order within each group
    from collections import OrderedDict
    by_type: dict[str, list] = OrderedDict()
    for r in results:
        st = r.get("source_type", "unknown")
        by_type.setdefault(st, []).append(r)

    if len(by_type) <= 1:
        return results  # only one source type — nothing to diversify

    # Round-robin: pick top chunk from each source type first (priority types first)
    selected_ids = set()
    diverse = []

    # Priority types go first
    for st in list(_PRIORITY_TYPES) + [t for t in by_type if t not in _PRIORITY_TYPES]:
        if st in by_type and by_type[st]:
            chunk = by_type[st][0]  # best chunk of this type
            cid = chunk.get("id")
            if cid not in selected_ids:
                diverse.append(chunk)
                selected_ids.add(cid)

    # Fill remaining slots with highest-similarity chunks across all types
    for r in results:
        if len(diverse) >= top_k * 6:
            break
        cid = r.get("id")
        if cid not in selected_ids:
            diverse.append(r)
            selected_ids.add(cid)

    type_counts = {}
    for r in diverse[:top_k]:
        st = r.get("source_type", "unknown")
        type_counts[st] = type_counts.get(st, 0) + 1
    logger.debug(f"Source diversity (top {top_k}): {type_counts}")

    return diverse


# ── Core retrieval ─────────────────────────────────────────────────────────────

def _run_search(cur, dense_vec, filters, params, limit):
    """Execute a single vector similarity search and return raw rows."""
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
    params_final = [dense_vec.tolist()] + params + [dense_vec.tolist(), limit]
    cur.execute(sql, params_final)
    return cur.fetchall()


def _rows_to_results(rows, sim_threshold):
    """Convert raw DB rows into result dicts, filtering by similarity."""
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
    return results


def retrieve(
    query: str,
    top_k: int = None,
    language_filter: str = None,
    source_type_filter: str = None,
    use_cache: bool = True,
) -> list:
    """
    Retrieve the most relevant document chunks for a query using dense search.
    Uses multi-source search to ensure all source types are represented.
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

    sim_threshold = float(os.getenv("SIMILARITY_THRESHOLD", str(_SIM_THRESHOLD_DEFAULT)))

    # Build base filters
    base_filters = []
    base_params = []
    if language_filter:
        base_filters.append("language = %s")
        base_params.append(language_filter)
    if source_type_filter:
        base_filters.append("source_type = %s")
        base_params.append(source_type_filter)

    conn = _get_db_connection()
    cur = conn.cursor()
    try:
        # ── Main search: global top results ───────────────────────────────
        main_rows = _run_search(cur, dense_vec, base_filters, base_params, top_k * 6)
        results = _rows_to_results(main_rows, sim_threshold)

        # ── Multi-source search: query each priority source type separately ──
        # This ensures textbook/manual/paper/sop chunks appear even when
        # qa_pair dominates by sheer volume.
        if not source_type_filter:
            seen_ids = {r["id"] for r in results}
            priority_added = 0
            for src_type in _PRIORITY_TYPES:
                extra_filters = base_filters + ["source_type = %s"]
                extra_params = base_params + [src_type]
                extra_rows = _run_search(cur, dense_vec, extra_filters, extra_params, 5)
                for r in _rows_to_results(extra_rows, sim_threshold):
                    if r["id"] not in seen_ids:
                        results.append(r)
                        seen_ids.add(r["id"])
                        priority_added += 1
            if priority_added:
                logger.info(f"Multi-source: added {priority_added} chunks from priority types")
    finally:
        cur.close()
        conn.close()

    # ── Reranking with guaranteed source-type diversity ─────────────────────
    # Split into priority (textbook/manual/paper/sop) and other (qa_pair).
    # Rerank each pool separately, then merge with guaranteed priority slots.
    priority_results = [r for r in results if r.get("source_type") in _PRIORITY_TYPES]
    other_results = [r for r in results if r.get("source_type") not in _PRIORITY_TYPES]

    try:
        from retrieval.reranker import rerank, RERANK_ENABLED
        if RERANK_ENABLED:
            # Rerank priority pool → keep best 3
            if len(priority_results) > 1:
                priority_results = rerank(query, priority_results, top_n=min(3, len(priority_results)))
            # Rerank other pool → keep best (top_k - priority slots)
            other_slots = max(top_k - len(priority_results), top_k // 2)
            if len(other_results) > 1:
                other_results = rerank(query, other_results, top_n=other_slots)
            else:
                other_results = other_results[:other_slots]

            # Merge: priority first (textbook/paper before qa_pair), then fill with others
            results = priority_results + other_results
            results = results[:top_k]
            type_counts = {}
            for r in results:
                st = r.get("source_type", "unknown")
                type_counts[st] = type_counts.get(st, 0) + 1
            logger.info(f"Reranked with diversity: {type_counts} ({len(results)} total)")
        else:
            results = _diversify_sources(results, top_k)
            results = results[:top_k]
    except Exception as e:
        logger.warning(f"Reranking skipped: {e}")
        results = _diversify_sources(results, top_k)
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

    # For non-English queries, translate to English for better embedding/retrieval.
    # The embedding model (bge-large-en-v1.5) only understands English.
    retrieval_query = query
    if language != "en":
        retrieval_query = _translate_query_to_english(query)

    results = retrieve(
        retrieval_query,
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
