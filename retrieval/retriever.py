"""
Glass Expert AI — Retrieval Module v3
=====================================
Hybrid search: dense vector + keyword matching + cross-encoder reranking.
Supports bilingual retrieval (English + Farsi) via bge-m3 multilingual embeddings.

Search pipeline:
  1. Dense vector search (bge-m3, 1024-dim) -> top_k * 4 candidates
  2. Keyword search (PostgreSQL ILIKE) -> top_k * 2 candidates (catches exact terms)
  3. Merge & deduplicate candidates
  4. Cross-encoder rerank (bge-reranker-v2-m3) -> final top_k results

Data source: documents_bgem3 table (247K chunks, bge-m3 multilingual embeddings)
"""

import os
import re
import sys
import json
import redis
import hashlib
import psycopg2
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from typing import Optional
from langdetect import detect, LangDetectException
from rank_bm25 import BM25Okapi

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.embedder import embed_query

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
TOP_K                  = int(os.getenv("RETRIEVAL_TOP_K", 30))
FINAL_TOP_K            = int(os.getenv("RETRIEVAL_FINAL_TOP_K", 8))
_SIM_THRESHOLD_DEFAULT = 0.30  # lowered to let reranker decide
REDIS_URL              = os.getenv("REDIS_URL", "redis://localhost:6379")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_PREFIX    = "glass_ai:semantic:"
SEMANTIC_CACHE_INDEX_KEY = "glass_ai:semantic:index"
REDIS_TTL = int(os.getenv("REDIS_TTL_SECONDS", "3600"))


# ── Database connection ────────────────────────────────────────────────────────
_pgvector_registered = set()

def _get_db_connection():
    """Get a pooled DB connection with pgvector extension registered."""
    from pgvector.psycopg2 import register_vector
    from api.database import get_db_conn
    conn = get_db_conn()
    conn_id = id(conn)
    if conn_id not in _pgvector_registered:
        register_vector(conn)
        _pgvector_registered.add(conn_id)
    return conn


def _return_db_connection(conn):
    from api.database import return_db_conn
    return_db_conn(conn)


# ── Redis cache connection ─────────────────────────────────────────────────────
def _get_redis():
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        return r
    except Exception:
        logger.warning("Redis not available -- caching disabled.")
        return None


# ── Language detection ─────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Detect query language, returning 'en' or 'fa'."""
    try:
        lang = detect(text)
        return "fa" if lang in ("fa", "ar") else "en"
    except LangDetectException:
        return "en"


# ── Query translation (for non-English -> English retrieval) ──────────────────
_GLOSSARY_CACHE = None

def _load_glossary() -> dict:
    """Load Persian -> English glass terminology glossary."""
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
            logger.warning("No OPENAI_API_KEY -- skipping query translation")
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
        logger.info(f"Translated query: '{query[:40]}...' -> '{translated[:80]}'")
        return translated
    except Exception as e:
        logger.warning(f"Query translation failed: {e}")
        return query


# ── Cache helpers ──────────────────────────────────────────────────────────────
def _semantic_cache_lookup(
    query_embedding: list,
    top_k: int,
    source_filter: str,
) -> Optional[list]:
    """
    Semantic cache lookup — returns cached results if a similar query
    (cosine similarity >= SEMANTIC_CACHE_THRESHOLD) was previously cached.
    Falls back gracefully if Redis unavailable.
    """
    r = _get_redis()
    if not r:
        return None
    try:
        import numpy as np
        # Load index of cached embeddings
        index_data = r.get(SEMANTIC_CACHE_INDEX_KEY)
        if not index_data:
            return None
        index = json.loads(index_data)
        q_vec = np.array(query_embedding, dtype=np.float32)
        best_sim = 0.0
        best_key = None
        for entry in index:
            if entry.get("top_k") != top_k:
                continue
            if entry.get("source_filter") != source_filter:
                continue
            cached_vec = np.array(entry["embedding"], dtype=np.float32)
            # Cosine similarity
            sim = float(np.dot(q_vec, cached_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(cached_vec) + 1e-10))
            if sim > best_sim:
                best_sim = sim
                best_key = entry.get("result_key")
        if best_sim >= SEMANTIC_CACHE_THRESHOLD and best_key:
            data = r.get(best_key)
            if data:
                logger.debug(f"Semantic cache hit (similarity={best_sim:.4f})")
                return json.loads(data)
        return None
    except Exception as e:
        logger.debug(f"Semantic cache lookup error: {e}")
        return None


def _semantic_cache_store(
    query: str,
    query_embedding: list,
    top_k: int,
    source_filter: str,
    results: list,
) -> None:
    """Store results in semantic cache with embedding for future similarity lookup."""
    r = _get_redis()
    if not r:
        return
    try:
        import numpy as np
        result_key = f"{SEMANTIC_CACHE_PREFIX}result:{hashlib.md5(query.encode()).hexdigest()}"
        r.setex(result_key, REDIS_TTL, json.dumps(results))
        # Update index
        index_data = r.get(SEMANTIC_CACHE_INDEX_KEY)
        index = json.loads(index_data) if index_data else []
        # Normalize embedding for storage (keep small — only first 128 dims for comparison)
        vec = np.array(query_embedding, dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        index.append({
            "query": query[:100],
            "embedding": vec.tolist()[:128],  # Store subset to keep index small
            "result_key": result_key,
            "top_k": top_k,
            "source_filter": source_filter,
        })
        # Keep index max 500 entries
        if len(index) > 500:
            index = index[-500:]
        r.setex(SEMANTIC_CACHE_INDEX_KEY, REDIS_TTL * 24, json.dumps(index))
        logger.debug(f"Semantic cache stored ({len(index)} entries in index)")
    except Exception as e:
        logger.debug(f"Semantic cache store error: {e}")

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


# ── Keyword extraction for hybrid search ─────────────────────────────────────
def _extract_keywords(query: str) -> list:
    """Extract meaningful keywords from a query for keyword search."""
    stopwords = {
        "what", "how", "why", "when", "where", "which", "who", "does", "do",
        "is", "are", "was", "were", "the", "a", "an", "of", "in", "for",
        "to", "and", "or", "with", "about", "from", "that", "this", "can",
        "tell", "me", "explain", "describe", "provide", "give", "its",
        "their", "they", "it", "be", "been", "being", "have", "has", "had",
    }
    # Keep chemical formulas intact (e.g. SiO2, Na2O, B2O3)
    words = re.findall(r'[A-Za-z][A-Za-z0-9]*(?:[.-][A-Za-z0-9]+)*', query)
    keywords = []
    for w in words:
        # Chemical formula detection: has both letters and numbers
        if re.match(r'^[A-Z][a-z]?\d', w):
            keywords.append(w)  # Keep original case for chemical formulas
        elif w.lower() not in stopwords and len(w) > 2:
            keywords.append(w.lower())
    return keywords


def _bm25_search(
    conn,
    query: str,
    language_filter: str = None,
    source_type_filter: str = None,
    limit: int = 20,
) -> list:
    """
    BM25 sparse search using rank_bm25 (Okapi BM25).
    Fetches candidate documents from Postgres then scores with BM25.
    More accurate than ILIKE for chemical formulas (SiO2, Na2O, B2O3)
    because BM25 accounts for term frequency and document length normalization.
    """
    cur = conn.cursor()
    try:
        filters = []
        params = []
        if language_filter:
            filters.append("language = %s")
            params.append(language_filter)
        if source_type_filter:
            filters.append("source_type = %s")
            params.append(source_type_filter)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""

        # Fetch candidate pool for BM25 scoring
        sql = f"""
            SELECT id::text, title, source_type, language, content, metadata
            FROM documents_bgem3
            {where}
            LIMIT %s
        """
        params.append(limit * 10)
        cur.execute(sql, params)
        rows = cur.fetchall()

        if not rows:
            return []

        # Tokenize corpus for BM25
        tokenized_corpus = [
            (row[4] or "").lower().split()
            for row in rows
        ]

        # Tokenize query
        tokenized_query = query.lower().split()

        # BM25 Okapi scoring
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Build results with BM25 scores
        scored = []
        for i, row in enumerate(rows):
            if scores[i] > 0:
                scored.append({
                    "id":          row[0],
                    "title":       row[1],
                    "source_type": row[2],
                    "language":    row[3],
                    "content":     row[4],
                    "metadata":    row[5] if isinstance(row[5], dict) else {},
                    "similarity":  0.0,
                    "bm25_score":  float(scores[i]),
                    "_source":     "bm25",
                })

        # Sort by BM25 score descending
        scored.sort(key=lambda r: r["bm25_score"], reverse=True)
        return scored[:limit]

    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        return []
    finally:
        cur.close()


# ── Core retrieval ─────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_k: int = None,
    language_filter: str = None,
    source_type_filter: str = None,
    use_cache: bool = True,
) -> list:
    """
    Hybrid retrieval: dense vector search + keyword search + cross-encoder reranking.

    Pipeline:
      1. Dense search -> top_k * 4 candidates (semantic similarity)
      2. Keyword search -> top_k * 2 candidates (exact term matching)
      3. Merge & deduplicate
      4. Cross-encoder rerank -> final top_k
    """
    top_k = top_k or FINAL_TOP_K

    # Check cache first
# Generate query embedding first (needed for semantic cache)
    logger.debug(f"Generating embedding for query: '{query[:80]}...'")
    query_emb = embed_query(query)
    dense_vec = query_emb["dense"]

    # Check exact cache first
    if use_cache:
        cached = _get_cached(query, top_k, str(language_filter))
        if cached:
            return cached
        # Semantic cache — similar query above threshold
        sem_cached = _semantic_cache_lookup(
            dense_vec.tolist(), top_k, str(source_type_filter)
        )
        if sem_cached is not None:
            return sem_cached

    conn = _get_db_connection()

    try:
        # ── Step 1: Dense vector search ───────────────────────────────────────
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
            FROM documents_bgem3
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

        dense_limit = top_k * 4
        params_final = [dense_vec.tolist()] + params + [dense_vec.tolist(), dense_limit]

        cur = conn.cursor()
        try:
            cur.execute(sql, params_final)
            rows = cur.fetchall()
        finally:
            cur.close()

        sim_threshold = float(os.getenv("SIMILARITY_THRESHOLD", str(_SIM_THRESHOLD_DEFAULT)))
        dense_results = []
        seen_ids = set()
        for row in rows:
            similarity = float(row[6])
            if similarity < sim_threshold:
                continue
            doc_id = row[0]
            seen_ids.add(doc_id)
            dense_results.append({
                "id":          doc_id,
                "title":       row[1],
                "source_type": row[2],
                "language":    row[3],
                "content":     row[4],
                "metadata":    row[5] if isinstance(row[5], dict) else {},
                "similarity":  round(similarity, 4),
                "_source":     "dense",
            })

        logger.debug(f"Dense search: {len(dense_results)} candidates (threshold={sim_threshold})")

        # ── Step 2: BM25 sparse search ────────────────────────────────────────────
        keywords = _extract_keywords(query)
        keyword_results = []
        if keywords:
            keyword_results = _bm25_search(
            conn, query,
            language_filter=language_filter,
            source_type_filter=source_type_filter,
            limit=top_k * 2,
        )
            # Deduplicate: only add keyword results not already in dense results
            new_keyword = [r for r in keyword_results if r["id"] not in seen_ids]
            logger.debug(f"Keyword search: {len(keyword_results)} total, {len(new_keyword)} new")
            keyword_results = new_keyword

        # ── Step 3: Reciprocal Rank Fusion (RRF) ──────────────────────────────
        # score = Σ 1/(k + rank_i) where k=60
        RRF_K = 60
        rrf_scores: dict = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_id]["score"] += 1.0 / (RRF_K + rank + 1)

        for rank, doc in enumerate(keyword_results):
            doc_id = doc["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_id]["score"] += 1.0 / (RRF_K + rank + 1)

        all_candidates = [
            entry["doc"]
            for entry in sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        ]
        logger.info(
            f"Hybrid RRF: {len(dense_results)} dense + {len(keyword_results)} BM25 "
            f"= {len(all_candidates)} candidates after RRF (k={RRF_K})"
        )

        # ── Step 4: Reranking (cross-encoder) ─────────────────────────────────
        try:
            from retrieval.reranker import rerank, RERANK_ENABLED
            if RERANK_ENABLED and len(all_candidates) > 1:
                results = rerank(query, all_candidates, top_n=top_k)
                logger.debug(f"Reranking applied: {len(results)} chunks after rerank")
            else:
                results = all_candidates[:top_k]
        except Exception as e:
            logger.warning(f"Reranking skipped: {e}")
            results = all_candidates[:top_k]

        # Clean up internal fields
        for r in results:
            r.pop("_source", None)
            r.pop("keyword_score", None)

# Cache the results
        if use_cache and results:
            _set_cache(query, top_k, str(language_filter), results)
            _semantic_cache_store(
                query=query,
                query_embedding=dense_vec.tolist(),
                top_k=top_k,
                source_filter=str(source_type_filter),
                results=results,
            )

        logger.info(f"Retrieved {len(results)} chunks for query")
        return results

    finally:
        _return_db_connection(conn)


def retrieve_with_auto_language(
    query: str,
    top_k: int = None,
    source_type_filter: str = None,
    language_override: str = None,
) -> tuple:
    """
    Retrieve results with automatic language detection and bilingual fallback.
    Returns (results, detected_language).
    """
    if language_override and language_override != "auto":
        language = language_override
    else:
        language = detect_language(query)
    logger.info(f"Detected query language: {language.upper()}")

    # bge-m3 is multilingual — Farsi queries are embedded directly without translation.
    # The embedding model understands both English and Farsi natively.
    # Translation is still available as optional boost for edge cases.
    retrieval_query = query
    if language != "en":
        # Try glossary-based enhancement: append English terms for domain-specific Farsi words
        glossary = _load_glossary()
        hints = [en_term for fa_term, en_term in glossary.items() if fa_term in query]
        if hints:
            retrieval_query = f"{query} ({', '.join(hints[:5])})"
            logger.info(f"Glossary-enhanced query: '{retrieval_query[:80]}'")

    # Search ALL languages — bge-m3 multilingual embeddings enable cross-lingual matching.
    # A Farsi query will naturally match relevant English chunks and vice versa.
    results = retrieve(
        retrieval_query,
        top_k=top_k,
        language_filter=None,  # search all languages — English corpus has the best coverage
        source_type_filter=source_type_filter,
    )

    return results, language


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English, ~2 for CJK/Farsi."""
    return len(text) // 3  # conservative estimate


def _smart_truncate(text: str, max_chars: int = 1500) -> str:
    """Truncate a chunk intelligently — keep first and last paragraphs,
    trim the middle. Preserves the most relevant info (usually at boundaries)."""
    if len(text) <= max_chars:
        return text

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 2:
        # Single or two paragraphs: just hard-truncate with ellipsis
        return text[:max_chars - 20].rsplit(" ", 1)[0] + "\n[...truncated...]"

    # Keep first paragraph(s) and last paragraph, trim middle
    head = paragraphs[0]
    tail = paragraphs[-1]

    # Fill remaining budget from middle paragraphs
    budget = max_chars - len(head) - len(tail) - 30  # 30 for separator
    middle_parts = []
    for p in paragraphs[1:-1]:
        if budget <= 0:
            break
        if len(p) <= budget:
            middle_parts.append(p)
            budget -= len(p)
        else:
            # Partial middle paragraph
            middle_parts.append(p[:budget].rsplit(" ", 1)[0] + "...")
            break

    parts = [head]
    if middle_parts:
        parts.extend(middle_parts)
    else:
        parts.append("[...truncated...]")
    parts.append(tail)
    return "\n\n".join(parts)


def format_context_for_llm(
    results: list,
    max_total_tokens: int = 3000,
    max_chunk_chars: int = 1500,
) -> str:
    """
    Format retrieved chunks into a structured context block for the LLM prompt.

    Smart token budgeting:
      - Each chunk is truncated to max_chunk_chars (~500 tokens)
      - Total context capped at max_total_tokens (~12K chars)
      - Higher-relevance chunks get more space
      - Enables fitting 6-8 sources in a 4K context window
    """
    if not results:
        return "No relevant information found in the knowledge base."

    max_total_chars = max_total_tokens * 4  # ~4 chars per token
    context_parts = ["KNOWLEDGE BASE CONTEXT:", "=" * 50]
    total_chars = 60  # header overhead

    for i, result in enumerate(results, 1):
        # Higher-relevance chunks get more space
        similarity = result.get("similarity", 0.5)
        rerank = result.get("rerank_score", 0)
        # Top results (rerank > 0.5 or sim > 0.7) get full budget, others get less
        if rerank > 0.5 or similarity > 0.7:
            chunk_budget = max_chunk_chars
        elif i <= 3:
            chunk_budget = max_chunk_chars  # always give top 3 full budget
        else:
            chunk_budget = max_chunk_chars // 2  # lower-ranked chunks get half

        content = _smart_truncate(result.get("content", ""), chunk_budget)

        header = (
            f"\n[Source {i}] {result['title']} "
            f"(Type: {result['source_type']} | "
            f"Language: {result['language'].upper()} | "
            f"Relevance: {result['similarity']:.0%})"
        )
        chunk_text = header + "\n" + content + "\n" + "-" * 40

        # Check if adding this chunk would exceed budget
        if total_chars + len(chunk_text) > max_total_chars and i > 3:
            # Always include at least 3 sources
            logger.debug(f"Context budget reached at source {i}, stopping")
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n".join(context_parts)


# Alias used by query router
format_context = format_context_for_llm