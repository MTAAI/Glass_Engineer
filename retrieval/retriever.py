"""
Glass Expert AI — Retrieval Module v3
=======================================
Search pipeline:
  1. bge-m3 embed query → dense vector + sparse lexical weights
  2. Dense ANN search (pgvector HNSW)         → top_k × 4 candidates
  3. Sparse BM25-style search (bge-m3 weights) → top_k × 2 candidates
  4. Reciprocal Rank Fusion (RRF)              → merged ranked list
  5. bge-reranker-v2-m3 cross-encoder          → precise top_k results
  6. Redis cache                               → 1h TTL on final results

Improvements vs both branches:
  - bge-m3 sparse weights used for retrieval (not thrown away)
  - RRF fusion outperforms simple concatenation + dedup
  - ILIKE keyword search retained as fallback when sparse weights absent
  - Deduplication by chunk ID (not title)
  - Per-source cap of 2 chunks for diversity
  - Farsi glossary translation support (Engineer Z's pattern)
  - Works with both psycopg2 ThreadedPool and asyncpg pool
  - Structured logging with loguru
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.embedder import embed_query

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
TOP_K                  = int(os.getenv("RETRIEVAL_TOP_K",       "30"))
FINAL_TOP_K            = int(os.getenv("RETRIEVAL_FINAL_TOP_K", "8"))
SIM_THRESHOLD          = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))
REDIS_URL              = os.getenv("REDIS_URL",         "redis://localhost:6379")
REDIS_TTL              = int(os.getenv("REDIS_TTL_SECONDS", "3600"))  # 1h
RRF_K                  = 60   # RRF constant — standard value


# ── DB connection (compatible with both pool patterns) ─────────────────────────
def _get_conn():
    """
    Get a database connection.
    Tries the new asyncpg-aware pool first, then Engineer Z's ThreadedPool,
    then falls back to a raw psycopg2 connection.
    """
    try:
        from api.database import get_db_conn
        from pgvector.psycopg2 import register_vector
        conn = get_db_conn()
        register_vector(conn)
        return conn, True   # (conn, pooled)
    except Exception:
        pass

    import psycopg2
    from pgvector.psycopg2 import register_vector
    conn = psycopg2.connect(os.getenv(
        "DATABASE_URL",
        "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai"
    ))
    register_vector(conn)
    return conn, False


def _return_conn(conn, pooled: bool) -> None:
    if pooled:
        try:
            from api.database import return_db_conn
            return_db_conn(conn)
            return
        except Exception:
            pass
    conn.close()


# ── Redis cache ────────────────────────────────────────────────────────────────
def _get_redis():
    try:
        import redis
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


def _cache_key(query: str, top_k: int, lang: str, src: str) -> str:
    raw = f"{query}|{top_k}|{lang}|{src}"
    return f"glass_ai:retrieval:{hashlib.md5(raw.encode()).hexdigest()}"


def _get_cached(key: str) -> Optional[list]:
    r = _get_redis()
    if not r:
        return None
    try:
        data = r.get(key)
        if data:
            logger.debug("Cache hit")
            return json.loads(data)
    except Exception:
        pass
    return None


def _set_cached(key: str, results: list) -> None:
    r = _get_redis()
    if not r or not results:
        return
    try:
        r.setex(key, REDIS_TTL, json.dumps(results))
    except Exception:
        pass


# ── Language detection ─────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return "fa" if lang in ("fa", "ar") else "en"
    except LangDetectException:
        return "en"


# ── Farsi → English query translation (Engineer Z's pattern) ──────────────────
_GLOSSARY: Optional[dict] = None

def _load_glossary() -> dict:
    global _GLOSSARY
    if _GLOSSARY is not None:
        return _GLOSSARY
    path = Path(__file__).parent / "glossary_fa_en.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _GLOSSARY = data.get("terms", {})
        logger.info(f"Loaded {len(_GLOSSARY)} Farsi glossary terms")
    else:
        _GLOSSARY = {}
    return _GLOSSARY


def _translate_to_english(query: str) -> str:
    """
    Translate a Farsi query to English using:
      1. Domain glossary lookup (fast, free, glass-specific)
      2. OpenAI gpt-4o-mini (if API key set)
    Returns the original query if translation fails.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.debug("No OPENAI_API_KEY — skipping Farsi translation")
            return query

        glossary = _load_glossary()
        hints = [f"{fa}={en}" for fa, en in glossary.items() if fa in query]
        hint_block = ""
        if hints:
            hint_block = "\n\nDomain terms:\n" + "\n".join(hints[:10])

        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the following glass science query to English. "
                        f"Return ONLY the English translation.{hint_block}"
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=200,
            temperature=0,
        )
        translated = resp.choices[0].message.content.strip()
        logger.info(f"Farsi query translated: '{query[:40]}' → '{translated[:80]}'")
        return translated
    except Exception as e:
        logger.warning(f"Query translation failed: {e}")
        return query


# ── Keyword extraction (fallback when sparse weights absent) ───────────────────
_STOPWORDS = {
    "what", "how", "why", "when", "where", "which", "who", "does", "is",
    "are", "was", "were", "the", "a", "an", "of", "in", "for", "to", "and",
    "or", "with", "about", "from", "that", "this", "can", "tell", "explain",
    "describe", "provide", "give", "its", "they", "it", "be", "been", "have",
    "has", "had",
}

def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords; preserve chemical formulas like SiO2, Na2O."""
    words = re.findall(r'[A-Za-z][A-Za-z0-9]*(?:[.-][A-Za-z0-9]+)*', query)
    keywords = []
    for w in words:
        if re.match(r'^[A-Z][a-z]?\d', w):   # chemical formula — keep original case
            keywords.append(w)
        elif w.lower() not in _STOPWORDS and len(w) > 2:
            keywords.append(w.lower())
    return keywords


# ── RRF fusion ─────────────────────────────────────────────────────────────────
def _rrf_fuse(
    dense_ranked: list[dict],
    sparse_ranked: list[dict],
) -> list[dict]:
    """
    Reciprocal Rank Fusion.
    RRF(d) = Σ 1 / (k + rank_i(d))    where k=60 (standard)
    """
    scores: dict[str, float]  = {}
    all_chunks: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_ranked, 1):
        cid = chunk["id"]
        scores[cid]     = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        all_chunks[cid] = chunk

    for rank, chunk in enumerate(sparse_ranked, 1):
        cid = chunk["id"]
        scores[cid]     = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        if cid not in all_chunks:
            all_chunks[cid] = chunk

    for cid, score in scores.items():
        all_chunks[cid]["rrf_score"] = score

    return sorted(all_chunks.values(), key=lambda c: c.get("rrf_score", 0), reverse=True)


# ── Dense search ───────────────────────────────────────────────────────────────
def _dense_search(
    cur,
    dense_vec,
    limit: int,
    language_filter: Optional[str],
    source_type_filter: Optional[str],
) -> list[dict]:
    filters, params = [], [dense_vec.tolist()]
    idx = 2

    if language_filter:
        filters.append(f"language = %s")
        params.append(language_filter)
        idx += 1
    if source_type_filter and source_type_filter != "all":
        filters.append(f"source_type = %s")
        params.append(source_type_filter)
        idx += 1

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params += [dense_vec.tolist(), limit]

    sql = f"""
        SELECT id::text, title, source_type, language, content, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        {where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    cur.execute(sql, params)
    rows = cur.fetchall()

    results = []
    for row in rows:
        sim = float(row[6])
        if sim < SIM_THRESHOLD:
            continue
        results.append({
            "id":          row[0],
            "title":       row[1] or "",
            "source_type": row[2] or "unknown",
            "language":    row[3] or "en",
            "content":     row[4] or "",
            "metadata":    row[5] if isinstance(row[5], dict) else {},
            "similarity":  round(sim, 4),
            "_source":     "dense",
        })
    return results


# ── Sparse search (bge-m3 lexical weights) ────────────────────────────────────
def _sparse_search(
    cur,
    sparse_weights: dict[str, float],
    limit: int,
    language_filter: Optional[str],
    source_type_filter: Optional[str],
) -> list[dict]:
    """
    Build a tsquery from bge-m3 top tokens.
    Falls back to ILIKE if tsquery fails or weights are empty.
    """
    if not sparse_weights:
        return []

    # Top-10 tokens by weight (skip very short tokens)
    top_tokens = sorted(sparse_weights.items(), key=lambda x: x[1], reverse=True)
    tokens = [t for t, _ in top_tokens if len(t) > 2][:10]
    if not tokens:
        return []

    filters, params = [], []

    # Try tsvector first (faster, uses the GIN index)
    try:
        tsquery = " | ".join(tokens)
        base_cond = "to_tsvector('english', content) @@ to_tsquery('english', %s)"
        params.append(tsquery)

        if language_filter:
            filters.append("language = %s")
            params.append(language_filter)
        if source_type_filter and source_type_filter != "all":
            filters.append("source_type = %s")
            params.append(source_type_filter)

        extra = (" AND " + " AND ".join(filters)) if filters else ""
        params.append(limit)

        sql = f"""
            SELECT id::text, title, source_type, language, content, metadata,
                   ts_rank(to_tsvector('english', content),
                            to_tsquery('english', %s)) AS score
            FROM documents
            WHERE {base_cond}{extra}
            ORDER BY score DESC
            LIMIT %s
        """
        params.insert(0, tsquery)   # first %s in SELECT ts_rank(...)
        cur.execute(sql, params)

    except Exception:
        # Fallback: ILIKE on first 3 tokens
        cur.execute(
            """SELECT id::text, title, source_type, language, content, metadata, 0.5
               FROM documents
               WHERE content ILIKE %s OR title ILIKE %s
               LIMIT %s""",
            (f"%{tokens[0]}%", f"%{tokens[0]}%", limit)
        )

    rows = cur.fetchall()
    return [
        {
            "id":          row[0],
            "title":       row[1] or "",
            "source_type": row[2] or "unknown",
            "language":    row[3] or "en",
            "content":     row[4] or "",
            "metadata":    row[5] if isinstance(row[5], dict) else {},
            "similarity":  float(row[6]),
            "_source":     "sparse",
        }
        for row in rows
    ]


# ── Deduplication ──────────────────────────────────────────────────────────────
def _deduplicate(chunks: list[dict], final_top_k: int, max_per_source: int = 2) -> list[dict]:
    """Deduplicate by chunk ID; cap at max_per_source chunks per title."""
    seen_ids:     set[str]  = set()
    title_counts: dict[str, int] = {}
    result: list[dict]      = []

    for chunk in chunks:
        cid   = chunk["id"]
        title = chunk.get("title", "").strip().lower()

        if cid in seen_ids:
            continue
        if title_counts.get(title, 0) >= max_per_source:
            continue

        seen_ids.add(cid)
        title_counts[title] = title_counts.get(title, 0) + 1
        result.append(chunk)

        if len(result) >= final_top_k:
            break

    return result


# ── Context formatter ──────────────────────────────────────────────────────────
def _smart_truncate(text: str, max_chars: int = 1500) -> str:
    """Keep first + last paragraphs, trim middle to fit budget."""
    if len(text) <= max_chars:
        return text
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 2:
        return text[:max_chars - 20].rsplit(" ", 1)[0] + "\n[...truncated...]"
    head, tail = paras[0], paras[-1]
    budget = max_chars - len(head) - len(tail) - 30
    middle = []
    for p in paras[1:-1]:
        if budget <= 0:
            break
        if len(p) <= budget:
            middle.append(p)
            budget -= len(p)
        else:
            middle.append(p[:budget].rsplit(" ", 1)[0] + "...")
            break
    parts = [head] + (middle or ["[...truncated...]"]) + [tail]
    return "\n\n".join(parts)


def format_context(
    results: list[dict],
    max_total_tokens: int = 3000,
    max_chunk_chars:  int = 1500,
) -> str:
    """Format retrieved chunks into a structured context block for the LLM."""
    if not results:
        return "No relevant information found in the knowledge base."

    max_total_chars = max_total_tokens * 4
    parts = ["KNOWLEDGE BASE CONTEXT:", "=" * 60]
    used  = 60

    for i, r in enumerate(results, 1):
        score   = r.get("rerank_score") or r.get("rrf_score") or r.get("similarity", 0)
        budget  = max_chunk_chars if (score > 0.5 or i <= 3) else max_chunk_chars // 2
        content = _smart_truncate(r.get("content", ""), budget)

        header = (
            f"\n[Source {i}] {r['title']}\n"
            f"Type: {r['source_type']} | Lang: {r['language'].upper()} | "
            f"Score: {score:.3f}"
        )
        block = header + "\n" + content + "\n" + "-" * 40

        if used + len(block) > max_total_chars and i > 3:
            logger.debug(f"Context budget reached at source {i}")
            break

        parts.append(block)
        used += len(block)

    return "\n".join(parts)


# ── Core retrieval ─────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_k: int = None,
    language_filter: Optional[str] = None,
    source_type_filter: Optional[str] = None,
    use_cache: bool = True,
) -> list[dict]:
    """
    Full hybrid retrieval:
      bge-m3 embed → dense search + sparse search → RRF → rerank → dedup → cache

    Returns list of chunk dicts with keys:
      id, title, source_type, language, content, metadata,
      similarity, rerank_score (if reranked), rrf_score
    """
    final_top_k = top_k or FINAL_TOP_K
    cache_key   = _cache_key(query, final_top_k, str(language_filter), str(source_type_filter))

    if use_cache:
        cached = _get_cached(cache_key)
        if cached:
            return cached

    # ── 1. Embed query ─────────────────────────────────────────────────────────
    embedding = embed_query(query)
    dense_vec = embedding["dense"]
    sparse_weights: dict = embedding.get("sparse", {})

    conn, pooled = _get_conn()
    cur = conn.cursor()

    try:
        # ── 2. Dense search ────────────────────────────────────────────────────
        dense_results = _dense_search(
            cur, dense_vec,
            limit=final_top_k * 4,
            language_filter=language_filter,
            source_type_filter=source_type_filter,
        )
        logger.debug(f"Dense: {len(dense_results)} candidates")

        # ── 3. Sparse search ───────────────────────────────────────────────────
        if sparse_weights:
            sparse_results = _sparse_search(
                cur, sparse_weights,
                limit=final_top_k * 2,
                language_filter=language_filter,
                source_type_filter=source_type_filter,
            )
        else:
            # Fallback: keyword search using extracted terms
            from retrieval.retriever import _extract_keywords
            keywords = _extract_keywords(query)
            sparse_results = []
            if keywords:
                try:
                    kw_cond = " OR ".join(
                        ["(content ILIKE %s OR title ILIKE %s)"] * min(len(keywords), 6)
                    )
                    kw_params = []
                    for kw in keywords[:6]:
                        kw_params.extend([f"%{kw}%", f"%{kw}%"])
                    kw_params.append(final_top_k * 2)
                    cur.execute(
                        f"SELECT id::text, title, source_type, language, content, metadata "
                        f"FROM documents WHERE {kw_cond} LIMIT %s",
                        kw_params,
                    )
                    for row in cur.fetchall():
                        sparse_results.append({
                            "id": row[0], "title": row[1] or "", "source_type": row[2] or "unknown",
                            "language": row[3] or "en", "content": row[4] or "",
                            "metadata": row[5] if isinstance(row[5], dict) else {},
                            "similarity": 0.5, "_source": "keyword",
                        })
                except Exception as e:
                    logger.warning(f"Keyword fallback failed: {e}")

        logger.debug(f"Sparse/keyword: {len(sparse_results)} candidates")

    finally:
        cur.close()
        _return_conn(conn, pooled)

    # ── 4. RRF fusion ──────────────────────────────────────────────────────────
    fused = _rrf_fuse(dense_results, sparse_results)
    logger.debug(f"After RRF: {len(fused)} unique candidates")

    # ── 5. Cross-encoder reranking ─────────────────────────────────────────────
    try:
        from retrieval.reranker import rerank, RERANK_ENABLED
        if RERANK_ENABLED and len(fused) > 1:
            fused = rerank(query, fused, top_n=min(final_top_k * 2, len(fused)))
            logger.debug(f"After rerank: {len(fused)}")
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")

    # ── 6. Dedup + truncate ────────────────────────────────────────────────────
    results = _deduplicate(fused, final_top_k)

    # Clean internal fields
    for r in results:
        r.pop("_source", None)

    logger.info(
        f"Retrieved {len(results)} chunks | "
        f"dense={len(dense_results)} sparse={len(sparse_results)}"
    )

    # ── 7. Cache ───────────────────────────────────────────────────────────────
    if use_cache and results:
        _set_cached(cache_key, results)

    return results


def retrieve_with_auto_language(
    query: str,
    top_k: int = None,
    source_type_filter: Optional[str] = None,
    language_override: Optional[str] = None,
) -> tuple[list[dict], str]:
    """
    Retrieve with automatic language detection and Farsi translation.
    Returns (chunks, detected_language).

    bge-m3 handles Farsi natively in the embedding space, but translating
    the query to English also improves recall on English-dominant corpus.
    Both approaches are attempted via the translation layer.
    """
    language = (
        language_override
        if language_override and language_override != "auto"
        else detect_language(query)
    )
    logger.info(f"Language detected: {language.upper()}")

    # For Farsi queries: translate to English for better retrieval on
    # the English-dominant corpus, then re-embed the translated query
    retrieval_query = query
    if language == "fa":
        retrieval_query = _translate_to_english(query)

    results = retrieve(
        query=retrieval_query,
        top_k=top_k,
        language_filter=None,      # search across all languages
        source_type_filter=source_type_filter,
    )
    return results, language