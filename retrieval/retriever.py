"""
Glass Expert AI — Retrieval Module v4 (Master)
===============================================
Hybrid search: dense vector + BM25 Okapi + RRF fusion + cross-encoder reranking.
Supports bilingual retrieval (English + Farsi) via bge-m3 multilingual embeddings.

Search pipeline:
  1. Dense vector search (bge-m3, 1024-dim) → top_k * 4 candidates
  2. BM25 Okapi search (rank_bm25) → top_k * 2 candidates (term frequency matching)
  3. Reciprocal Rank Fusion (RRF, k=60) to merge dense + BM25 ranked lists
  4. Cross-encoder rerank (bge-reranker-v2-m3) → final top_k results

Caching:
  - Exact cache: MD5 hash of (query, top_k, language_filter) → Redis
  - Semantic cache: cosine similarity on first 128 dims of query embedding
    (threshold 0.92 = near-identical questions hit cache without exact match)

Data source: documents_bgem3 table (247K chunks, bge-m3 multilingual embeddings)
"""

import os
import re
import sys
import json
import time
import redis
import hashlib
import numpy as np
import psycopg2
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from rank_bm25 import BM25Okapi

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.embedder import embed_query

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
TOP_K                  = int(os.getenv("RETRIEVAL_TOP_K", 20))
FINAL_TOP_K            = int(os.getenv("RETRIEVAL_FINAL_TOP_K", 6))
_SIM_THRESHOLD_DEFAULT = 0.35  # raised — bge-m3 scores are well-calibrated, 0.30 lets noise through
REDIS_URL              = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_TTL              = int(os.getenv("REDIS_TTL_SECONDS", 86400))

# Semantic cache config
SEMANTIC_CACHE_THRESHOLD   = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_DIM         = 128   # use first 128 dims for fast comparison
SEMANTIC_CACHE_MAX_ENTRIES = 500   # max cached queries in Redis
SEMANTIC_CACHE_PREFIX      = "glass_ai:semantic:"
SEMANTIC_CACHE_INDEX_KEY   = "glass_ai:semantic:index"

# RRF config
RRF_K = int(os.getenv("RRF_K", "60"))  # standard RRF constant


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
        # Set HNSW search ef parameter — controls recall vs speed tradeoff
        # Higher ef = better recall but slower. 100 is good for 247K docs.
        cur = conn.cursor()
        cur.execute("SET hnsw.ef_search = 100")
        cur.close()
        _pgvector_registered.add(conn_id)
    return conn


def _return_db_connection(conn):
    from api.database import return_db_conn
    return_db_conn(conn)


# ── Redis cache connection ─────────────────────────────────────────────────────
_redis_conn = None

def _get_redis():
    """Get or create a cached Redis connection (singleton)."""
    global _redis_conn
    if _redis_conn is not None:
        try:
            _redis_conn.ping()
            return _redis_conn
        except Exception:
            _redis_conn = None
    try:
        _redis_conn = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_conn.ping()
        return _redis_conn
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
            logger.warning("No OPENAI_API_KEY -- skipping query translation")
            return query

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


# ── Exact cache helpers ───────────────────────────────────────────────────────
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
        logger.debug("Exact cache hit for query")
        return json.loads(cached)
    return None


def _set_cache(query: str, top_k: int, language_filter: str, results: list):
    r = _get_redis()
    if not r:
        return
    key = _cache_key(query, top_k, language_filter)
    r.setex(key, REDIS_TTL, json.dumps(results))


# ── Semantic cache ────────────────────────────────────────────────────────────
def _semantic_cache_lookup(query_embedding, top_k: int, source_filter: str):
    """
    Semantic cache: find a cached result whose query embedding is
    cosine-similar (>= threshold) to the current query.
    Uses first 128 dims for speed. Returns cached results or None.
    """
    r = _get_redis()
    if not r:
        return None
    try:
        index_data = r.get(SEMANTIC_CACHE_INDEX_KEY)
        if not index_data:
            return None

        index = json.loads(index_data)
        q_vec = np.array(query_embedding[:SEMANTIC_CACHE_DIM], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return None

        best_sim = 0.0
        best_key = None

        for entry in index:
            if entry.get("top_k") != top_k:
                continue
            if entry.get("source_filter") != source_filter:
                continue
            cached_vec = np.array(entry["embedding"], dtype=np.float32)
            c_norm = np.linalg.norm(cached_vec)
            if c_norm == 0:
                continue
            sim = float(np.dot(q_vec, cached_vec) / (q_norm * c_norm))
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


def _semantic_cache_store(query: str, query_embedding, top_k: int, source_filter: str, results: list):
    """Store results in semantic cache with embedding for future similarity lookup."""
    r = _get_redis()
    if not r:
        return
    try:
        result_key = f"{SEMANTIC_CACHE_PREFIX}result:{hashlib.md5(query.encode()).hexdigest()}"
        r.setex(result_key, REDIS_TTL, json.dumps(results))

        # Load and update the index
        index_data = r.get(SEMANTIC_CACHE_INDEX_KEY)
        index = json.loads(index_data) if index_data else []

        # Normalize and truncate embedding to 128 dims
        vec = np.array(query_embedding[:SEMANTIC_CACHE_DIM], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        index.append({
            "query": query[:100],
            "embedding": vec.tolist(),
            "result_key": result_key,
            "top_k": top_k,
            "source_filter": source_filter,
        })

        # Keep index bounded
        if len(index) > SEMANTIC_CACHE_MAX_ENTRIES:
            index = index[-SEMANTIC_CACHE_MAX_ENTRIES:]

        r.setex(SEMANTIC_CACHE_INDEX_KEY, REDIS_TTL * 24, json.dumps(index))
        logger.debug(f"Semantic cache stored ({len(index)} entries in index)")
    except Exception as e:
        logger.debug(f"Semantic cache store error: {e}")


# ── Keyword extraction for BM25 (English only) ───────────────────────────────
def _extract_keywords(query: str) -> list:
    """Extract meaningful English keywords from a query for BM25 search.
    BM25 is only used for English queries — Farsi uses dense bge-m3 only."""
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
        if re.match(r'^[A-Z][a-z]?\d', w):
            keywords.append(w)  # Keep original case for chemical formulas
        elif w.lower() not in stopwords and len(w) > 2:
            keywords.append(w.lower())
    return keywords


# ── BM25 Okapi search (replaces old ILIKE keyword search) ────────────────────
def _bm25_search(
    conn,
    query: str,
    keywords: list,
    language_filter: str = None,
    source_type_filter: str = None,
    limit: int = 20,
) -> list:
    """
    BM25 Okapi sparse search using rank_bm25.

    Fetches candidate documents from Postgres via ILIKE (fast pre-filter),
    then scores with BM25 Okapi for proper term frequency / document length
    normalization. Much more accurate than raw ILIKE for:
      - Chemical formulas (SiO2, Na2O, B2O3)
      - Glass terminology (devitrification, annealing point)
      - Multi-word technical phrases
    """
    if not keywords:
        return []

    cur = conn.cursor()
    try:
        # Step 1: ILIKE pre-filter to get candidate pool for BM25 scoring
        # This is much faster than scoring the entire 247K corpus
        keyword_conditions = []
        params = []
        for kw in keywords[:8]:
            keyword_conditions.append("(content ILIKE %s OR title ILIKE %s)")
            pattern = f"%{kw}%"
            params.extend([pattern, pattern])

        keyword_where = " OR ".join(keyword_conditions)

        extra_filters = []
        if language_filter:
            extra_filters.append("language = %s")
            params.append(language_filter)
        if source_type_filter:
            extra_filters.append("source_type = %s")
            params.append(source_type_filter)

        extra_where = (" AND " + " AND ".join(extra_filters)) if extra_filters else ""

        sql = f"""
            SELECT id::text, title, source_type, language, content, metadata
            FROM documents_bgem3
            WHERE ({keyword_where}){extra_where}
            LIMIT %s
        """
        params.append(limit * 3)  # fetch 3x candidates (was 5x — smaller pool = faster BM25)

        cur.execute(sql, params)
        rows = cur.fetchall()

        if not rows:
            return []

        # Step 2: BM25 Okapi scoring on the candidate pool
        tokenized_corpus = [
            (row[4] or "").lower().split() for row in rows
        ]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Step 3: Build results with BM25 scores
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
                    "similarity":  0.0,  # placeholder — reranker will rescore
                    "bm25_score":  float(scores[i]),
                    "_source":     "bm25",
                })

        scored.sort(key=lambda r: r["bm25_score"], reverse=True)
        return scored[:limit]

    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        return []
    finally:
        cur.close()


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def _rrf_merge(dense_results: list, bm25_results: list) -> list:
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).
    score(doc) = sum( 1 / (k + rank_i) ) for each list where doc appears.

    RRF is robust to score scale differences between dense similarity and BM25 TF-IDF.
    k=60 is the standard constant from the original RRF paper (Cormack et al., 2009).
    """
    rrf_scores: dict = {}

    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
        rrf_scores[doc_id]["score"] += 1.0 / (RRF_K + rank + 1)

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
        rrf_scores[doc_id]["score"] += 1.0 / (RRF_K + rank + 1)

    # Sort by RRF score descending
    merged = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return [entry["doc"] for entry in merged]


# ── Core retrieval ────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_k: int = None,
    language_filter: str = None,
    source_type_filter: str = None,
    use_cache: bool = True,
) -> list:
    """
    Hybrid retrieval pipeline:
      1. Dense vector search (bge-m3) → top_k * 4 candidates
      2. BM25 Okapi search → top_k * 2 candidates
      3. RRF fusion to merge ranked lists
      4. Cross-encoder rerank → final top_k results

    Caching: exact MD5 → semantic cosine → full retrieval
    """
    top_k = top_k or FINAL_TOP_K

    # Generate query embedding (needed for both search and semantic cache)
    logger.debug(f"Generating embedding for query: '{query[:80]}...'")
    query_emb = embed_query(query)
    dense_vec = query_emb["dense"]

    # ── Cache lookup: exact → semantic ────────────────────────────────────
    if use_cache:
        cached = _get_cached(query, top_k, str(language_filter))
        if cached:
            return cached
        sem_cached = _semantic_cache_lookup(
            dense_vec.tolist(), top_k, str(source_type_filter)
        )
        if sem_cached is not None:
            return sem_cached

    conn = _get_db_connection()

    try:
        # ── Step 1: Two-pass dense vector search ──────────────────────────
        # Pass 1: Priority sources (textbook + qa_pair) — ensures fundamentals surface first
        # Pass 2: All sources — fills remaining slots with papers, manuals, etc.
        # This prevents niche research papers from drowning out textbook definitions.
        _PRIORITY_TYPES = ("textbook", "qa_pair")

        sim_threshold = float(os.getenv("SIMILARITY_THRESHOLD", str(_SIM_THRESHOLD_DEFAULT)))
        dense_results = []
        seen_ids = set()
        dense_limit = top_k * 3  # reduced from *4 — faster pgvector scan

        for pass_num, type_filter in enumerate((_PRIORITY_TYPES, None), 1):
            filters = []
            params = []

            if language_filter:
                filters.append("language = %s")
                params.append(language_filter)
            if source_type_filter:
                filters.append("source_type = %s")
                params.append(source_type_filter)
            elif type_filter:
                # Pass 1: only priority types
                placeholders = ",".join(["%s"] * len(type_filter))
                filters.append(f"source_type IN ({placeholders})")
                params.extend(type_filter)

            where_clause = ("WHERE " + " AND ".join(filters)) if filters else ""

            sql = f"""
                SELECT
                    id::text, title, source_type, language, content, metadata,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM documents_bgem3
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """

            pass_limit = dense_limit // 2 if type_filter else dense_limit
            params_final = [dense_vec.tolist()] + params + [dense_vec.tolist(), pass_limit]

            cur = conn.cursor()
            try:
                cur.execute(sql, params_final)
                rows = cur.fetchall()
            finally:
                cur.close()

            for row in rows:
                similarity = float(row[6])
                if similarity < sim_threshold:
                    continue
                doc_id = row[0]
                if doc_id in seen_ids:
                    continue
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

            logger.debug(f"Dense pass {pass_num}: {len(dense_results)} total candidates (threshold={sim_threshold})")

            # If pass 1 filled enough slots, skip pass 2
            if len(dense_results) >= dense_limit:
                break

        # ── Step 2: BM25 Okapi search (English only) ─────────────────────
        # BM25 = English keyword matching (SiO2, Na2O, glass terms).
        # Farsi queries skip BM25 — dense bge-m3 handles Farsi natively.
        query_lang = detect_language(query)
        keywords = _extract_keywords(query) if query_lang == "en" else []
        bm25_results = []
        if keywords:
            bm25_results = _bm25_search(
                conn, query, keywords,
                language_filter=language_filter,
                source_type_filter=source_type_filter,
                limit=top_k * 2,
            )
            # Deduplicate: only add BM25 results not already in dense results
            new_bm25 = [r for r in bm25_results if r["id"] not in seen_ids]
            logger.debug(f"BM25 search: {len(bm25_results)} total, {len(new_bm25)} new")
            bm25_results = new_bm25

        # ── Step 3: Reciprocal Rank Fusion ────────────────────────────────
        all_candidates = _rrf_merge(dense_results, bm25_results)
        logger.info(
            f"Hybrid RRF: {len(dense_results)} dense + {len(bm25_results)} BM25 "
            f"= {len(all_candidates)} candidates (k={RRF_K})"
        )

        # ── Step 4: Cross-encoder reranking ───────────────────────────────
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

        # ── Source type priority boost ──────────────────────────────────
        # Textbooks and qa_pairs are more reliable for general questions.
        # Papers get a small penalty to prevent niche research from drowning out fundamentals.
        _BOOST = {"textbook": 0.15, "qa_pair": 0.10, "standard": 0.08, "sop": 0.05, "manual": 0.02, "paper": -0.03}
        for r in results:
            stype = (r.get("source_type") or "").lower()
            boost = _BOOST.get(stype, 0.0)
            if boost:
                r["similarity"] = max(0.0, min(1.0, r.get("similarity", 0.5) + boost))
                if "rerank_score" in r:
                    r["rerank_score"] = max(0.0, min(1.0, r["rerank_score"] + boost))
        # Re-sort after boosting
        results.sort(
            key=lambda r: r.get("rerank_score", r.get("similarity", 0)),
            reverse=True,
        )

        # Clean up internal fields
        for r in results:
            r.pop("_source", None)
            r.pop("bm25_score", None)

        # ── Cache store: exact + semantic ─────────────────────────────────
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

    # bge-m3 is multilingual — Farsi queries are embedded directly.
    # The embedding model understands both English and Farsi natively.
    retrieval_query = query
    if language != "en":
        # Glossary-based enhancement: append English terms for domain-specific Farsi words
        glossary = _load_glossary()
        hints = [en_term for fa_term, en_term in glossary.items() if fa_term in query]
        if hints:
            retrieval_query = f"{query} ({', '.join(hints[:5])})"
            logger.info(f"Glossary-enhanced query: '{retrieval_query[:80]}'")

    # Bilingual retrieval strategy:
    # - English queries: search all languages (bge-m3 cross-lingual matching)
    # - Farsi queries: search all languages too (English corpus has best glass science coverage)
    #   but bge-m3 natively handles Farsi, so relevant Farsi docs will naturally rank higher
    # Note: we do NOT filter by language — the reranker handles relevance across languages.
    # This ensures Farsi queries can still access English-only technical content.
    results = retrieve(
        retrieval_query,
        top_k=top_k,
        language_filter=None,  # cross-lingual: let bge-m3 + reranker decide
        source_type_filter=source_type_filter,
    )

    return results, language


# ── Context formatting for LLM ────────────────────────────────────────────────
def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English, ~2 for CJK/Farsi."""
    return len(text) // 3  # conservative estimate


def _smart_truncate(text: str, max_chars: int = 1500) -> str:
    """Truncate a chunk intelligently — keep first and last paragraphs,
    trim the middle. Preserves the most relevant info (usually at boundaries)."""
    if len(text) <= max_chars:
        return text

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 2:
        return text[:max_chars - 20].rsplit(" ", 1)[0] + "\n[...truncated...]"

    head = paragraphs[0]
    tail = paragraphs[-1]

    budget = max_chars - len(head) - len(tail) - 30
    middle_parts = []
    for p in paragraphs[1:-1]:
        if budget <= 0:
            break
        if len(p) <= budget:
            middle_parts.append(p)
            budget -= len(p)
        else:
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
    max_total_tokens: int = 1800,
    max_chunk_chars: int = 1000,
) -> str:
    """
    Format retrieved chunks into a structured context block for the LLM prompt.

    Smart token budgeting:
      - Each chunk is truncated to max_chunk_chars (~300 tokens)
      - Total context capped at max_total_tokens (~8K chars)
      - Higher-relevance chunks get more space
      - Enables fitting 6-8 sources in a 4K context window
    """
    if not results:
        return "No relevant information found in the knowledge base."

    max_total_chars = max_total_tokens * 4  # ~4 chars per token
    context_parts = ["KNOWLEDGE BASE CONTEXT:", "=" * 50]
    total_chars = 60  # header overhead

    for i, result in enumerate(results, 1):
        similarity = result.get("similarity", 0.5)
        rerank = result.get("rerank_score", 0)
        # Top results get full budget, lower-ranked get half
        if rerank > 0.5 or similarity > 0.7:
            chunk_budget = max_chunk_chars
        elif i <= 3:
            chunk_budget = max_chunk_chars  # always give top 3 full budget
        else:
            chunk_budget = max_chunk_chars // 2

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
            logger.debug(f"Context budget reached at source {i}, stopping")
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n".join(context_parts)


# Alias used by query router
format_context = format_context_for_llm
