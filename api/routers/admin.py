"""
Glass Expert AI — Admin Control API
=====================================
Complete admin control over the system.
All endpoints require is_admin=True JWT check.

Endpoint groups:
  1. GET  /admin/health          — Component-level system health
  2. GET  /admin/stats           — System-wide statistics
  3. GET  /admin/analytics       — Query analytics (p50/p95/fallback rate)
  4. GET  /admin/users           — List all users with activity
  5. PATCH /admin/users/{id}     — Enable/disable user accounts
  6. GET  /admin/ingestion/log   — Ingestion history
  7. POST /admin/ingestion/reindex — Re-ingest failed documents
  8. GET  /admin/documents       — List indexed documents
  9. DELETE /admin/documents/{id} — Delete a document
  10. POST /admin/documents/{id}/reembed — Re-embed a document
  11. GET  /admin/cache/stats    — Semantic cache stats
  12. POST /admin/cache/flush    — Flush cache
  13. GET  /admin/rate-limits    — View rate limit config
  14. PATCH /admin/rate-limits   — Update rate limits
"""
import os
import json
import shutil
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from loguru import logger
from api.auth import require_admin, UserInToken
from api.database import get_db

router = APIRouter()


# ── 1. System Health ───────────────────────────────────────────────────────────
@router.get("/admin/health")
async def admin_health(user: UserInToken = Depends(require_admin)):
    """
    Component-level health check.
    Returns green/red per component for production monitoring.
    """
    health = {}

    # PostgreSQL
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        health["postgresql"] = {"status": "green", "message": "Connected"}
    except Exception as e:
        health["postgresql"] = {"status": "red", "message": str(e)}

    # Redis
    try:
        import redis as redis_lib
        r = redis_lib.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        r.ping()
        info = r.info("memory")
        health["redis"] = {
            "status": "green",
            "message": "Connected",
            "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
        }
    except Exception as e:
        health["redis"] = {"status": "red", "message": str(e)}

    # Embedding model
    try:
        from ingestion.embedder import _get_model
        model = _get_model()
        health["embedding_model"] = {
            "status": "green",
            "message": f"Loaded: {os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')}",
        }
    except Exception as e:
        health["embedding_model"] = {"status": "red", "message": str(e)}

    # LLM server
    try:
        import httpx
        llm_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        r = httpx.get(f"{llm_url.rstrip('/v1')}/health", timeout=3.0)
        health["llm_server"] = {
            "status": "green" if r.status_code == 200 else "yellow",
            "message": f"HTTP {r.status_code}",
        }
    except Exception as e:
        health["llm_server"] = {"status": "red", "message": f"Unavailable: {str(e)[:80]}"}

    # Disk usage
    try:
        disk = shutil.disk_usage("/")
        health["disk"] = {
            "status": "green" if disk.free > 10 * 1024**3 else "yellow",
            "total_gb": round(disk.total / 1024**3, 1),
            "used_gb": round(disk.used / 1024**3, 1),
            "free_gb": round(disk.free / 1024**3, 1),
            "used_pct": round(disk.used / disk.total * 100, 1),
        }
    except Exception as e:
        health["disk"] = {"status": "red", "message": str(e)}

    # Last ingestion
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT file_name, status, created_at
                FROM ingestion_log
                ORDER BY created_at DESC LIMIT 1
            """)
            row = cur.fetchone()
            cur.close()
        if row:
            health["last_ingestion"] = {
                "file": row[0],
                "status": row[1],
                "at": row[2].isoformat() if row[2] else None,
            }
        else:
            health["last_ingestion"] = {"status": "no ingestions yet"}
    except Exception as e:
        health["last_ingestion"] = {"status": "error", "message": str(e)}

    overall = "green" if all(
        v.get("status") == "green"
        for v in health.values()
        if isinstance(v, dict) and "status" in v
    ) else "degraded"

    return {"overall": overall, "components": health, "timestamp": datetime.utcnow().isoformat()}


# ── 2. System Stats ────────────────────────────────────────────────────────────
@router.get("/admin/stats")
async def get_dashboard_stats(user: UserInToken = Depends(require_admin)):
    """System-wide statistics for the admin dashboard."""
    with get_db() as conn:
        cur = conn.cursor()
        stats = {}

        cur.execute("SELECT count(*) FROM documents_bgem3")
        stats["total_chunks"] = cur.fetchone()[0]

        cur.execute("SELECT source_type, count(*) FROM documents_bgem3 GROUP BY source_type ORDER BY count(*) DESC")
        stats["chunks_by_type"] = [{"source_type": r[0], "count": r[1]} for r in cur.fetchall()]

        cur.execute("SELECT language, count(*) FROM documents_bgem3 GROUP BY language ORDER BY count(*) DESC")
        stats["chunks_by_language"] = [{"language": r[0], "count": r[1]} for r in cur.fetchall()]

        cur.execute("SELECT count(*) FROM users")
        stats["total_users"] = cur.fetchone()[0]

        cur.execute("""
            SELECT count(DISTINCT user_id) FROM chat_history
            WHERE created_at > now() - interval '7 days' AND user_id IS NOT NULL
        """)
        stats["active_users_7d"] = cur.fetchone()[0]

        cur.execute("SELECT count(DISTINCT session_id) FROM chat_history")
        stats["total_conversations"] = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM chat_history")
        stats["total_messages"] = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM chat_history WHERE created_at > now() - interval '7 days'")
        stats["messages_7d"] = cur.fetchone()[0]

        cur.execute("""
            SELECT count(*) as total,
                   count(*) FILTER (WHERE rating = 1) as positive,
                   count(*) FILTER (WHERE rating = -1) as negative
            FROM feedback
        """)
        row = cur.fetchone()
        stats["feedback"] = {
            "total": row[0], "positive": row[1], "negative": row[2],
            "satisfaction_pct": round(row[1] / row[0] * 100, 1) if row[0] > 0 else 0,
        }

        cur.execute("""
            SELECT metadata->>'model_used' as model, count(*) as cnt
            FROM chat_history
            WHERE role = 'assistant' AND metadata->>'model_used' IS NOT NULL
            GROUP BY model ORDER BY cnt DESC
        """)
        stats["model_usage"] = [{"model": r[0], "count": r[1]} for r in cur.fetchall()]

        cur.close()
        return stats


# ── 3. Query Analytics ─────────────────────────────────────────────────────────
@router.get("/admin/analytics")
async def get_query_analytics(user: UserInToken = Depends(require_admin)):
    """Query analytics — latency stats, fallback rate, slowest queries."""
    with get_db() as conn:
        cur = conn.cursor()
        analytics = {}

        cur.execute("""
            SELECT
                COUNT(*) as total_queries,
                ROUND(AVG(total_latency_ms)::numeric, 0) as avg_latency_ms,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_latency_ms)::numeric, 0) as p50_latency_ms,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_latency_ms)::numeric, 0) as p95_latency_ms,
                ROUND(AVG(retrieval_latency_ms)::numeric, 0) as avg_retrieval_ms,
                ROUND(AVG(top_rerank_score)::numeric, 4) as avg_rerank_score
            FROM query_analytics
            WHERE created_at > now() - interval '24 hours'
        """)
        row = cur.fetchone()
        analytics["last_24h"] = {
            "total_queries": row[0],
            "avg_latency_ms": float(row[1]) if row[1] else 0,
            "p50_latency_ms": float(row[2]) if row[2] else 0,
            "p95_latency_ms": float(row[3]) if row[3] else 0,
            "avg_retrieval_ms": float(row[4]) if row[4] else 0,
            "avg_rerank_score": float(row[5]) if row[5] else 0,
        }

        cur.execute("""
            SELECT DATE_TRUNC('hour', created_at) as hour, COUNT(*) as queries
            FROM query_analytics
            WHERE created_at > now() - interval '24 hours'
            GROUP BY hour ORDER BY hour DESC
        """)
        analytics["queries_per_hour"] = [
            {"hour": r[0].isoformat(), "queries": r[1]} for r in cur.fetchall()
        ]

        cur.execute("""
            SELECT COUNT(*) as total,
                   COUNT(*) FILTER (WHERE fallback_used = TRUE) as fallbacks
            FROM query_analytics
            WHERE created_at > now() - interval '24 hours'
        """)
        row = cur.fetchone()
        total = row[0] or 1
        analytics["fallback_rate_pct"] = round(row[1] / total * 100, 1)
        analytics["fallback_count"] = row[1]

        cur.execute("""
            SELECT fallback_reason, COUNT(*) as cnt
            FROM query_analytics
            WHERE fallback_used = TRUE AND created_at > now() - interval '24 hours'
            GROUP BY fallback_reason ORDER BY cnt DESC
        """)
        analytics["fallback_reasons"] = [{"reason": r[0], "count": r[1]} for r in cur.fetchall()]

        cur.execute("""
            SELECT language, COUNT(*) as cnt
            FROM query_analytics
            WHERE created_at > now() - interval '24 hours'
            GROUP BY language ORDER BY cnt DESC
        """)
        analytics["queries_by_language"] = [{"language": r[0], "count": r[1]} for r in cur.fetchall()]

        cur.execute("""
            SELECT query_text, total_latency_ms, retrieval_latency_ms, model_used, created_at
            FROM query_analytics
            WHERE created_at > now() - interval '24 hours'
            ORDER BY total_latency_ms DESC LIMIT 10
        """)
        analytics["slowest_queries"] = [
            {
                "query": r[0][:100], "total_ms": round(r[1], 0),
                "retrieval_ms": round(r[2], 0), "model": r[3],
                "time": r[4].isoformat(),
            }
            for r in cur.fetchall()
        ]

        cur.close()
        return analytics


# ── 4. User Management ─────────────────────────────────────────────────────────
@router.get("/admin/users")
async def list_users(user: UserInToken = Depends(require_admin)):
    """List all users with last active time and query count."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                u.id, u.email, u.full_name, u.role,
                u.is_active, u.created_at,
                MAX(ch.created_at) as last_active,
                COUNT(DISTINCT ch.session_id) as conversation_count,
                COUNT(ch.id) FILTER (WHERE ch.role = 'user') as query_count
            FROM users u
            LEFT JOIN chat_history ch ON ch.user_id = u.id::text
            GROUP BY u.id, u.email, u.full_name, u.role, u.is_active, u.created_at
            ORDER BY last_active DESC NULLS LAST
        """)
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "id": str(r[0]),
            "email": r[1],
            "full_name": r[2],
            "role": r[3],
            "is_active": r[4],
            "created_at": r[5].isoformat() if r[5] else None,
            "last_active": r[6].isoformat() if r[6] else None,
            "conversation_count": r[7],
            "query_count": r[8],
        }
        for r in rows
    ]


class UserUpdateRequest(BaseModel):
    is_active: Optional[bool] = None
    role: Optional[str] = None


@router.patch("/admin/users/{user_id}")
async def update_user(
    user_id: str,
    body: UserUpdateRequest,
    user: UserInToken = Depends(require_admin),
):
    """Enable/disable user accounts or change role."""
    updates = []
    params = []
    if body.is_active is not None:
        updates.append("is_active = %s")
        params.append(body.is_active)
    if body.role is not None:
        if body.role not in ("user", "admin"):
            raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")
        updates.append("role = %s")
        params.append(body.role)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    params.append(user_id)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = %s RETURNING id, email, is_active, role",
            params,
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()

    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": str(row[0]), "email": row[1], "is_active": row[2], "role": row[3]}


# ── 5. Ingestion Management ────────────────────────────────────────────────────
@router.get("/admin/ingestion/log")
async def get_ingestion_log(
    limit: int = 50,
    status: Optional[str] = None,
    user: UserInToken = Depends(require_admin),
):
    """View ingestion history with successes, failures, chunk counts."""
    with get_db() as conn:
        cur = conn.cursor()
        where = "WHERE status = %s" if status else ""
        params = [status, limit] if status else [limit]
        cur.execute(f"""
            SELECT file_name, file_path, source_type, language,
                   chunk_count, status, error_message, created_at
            FROM ingestion_log
            {where}
            ORDER BY created_at DESC
            LIMIT %s
        """, params)
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "file_name": r[0], "file_path": r[1], "source_type": r[2],
            "language": r[3], "chunk_count": r[4], "status": r[5],
            "error_message": r[6],
            "created_at": r[7].isoformat() if r[7] else None,
        }
        for r in rows
    ]


@router.post("/admin/ingestion/reindex")
async def reindex_failed(user: UserInToken = Depends(require_admin)):
    """Get list of failed ingestions that can be re-triggered."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT file_name, file_path, source_type, error_message, created_at
            FROM ingestion_log
            WHERE status = 'failed'
            ORDER BY created_at DESC
            LIMIT 50
        """)
        rows = cur.fetchall()
        cur.close()

    failed = [
        {
            "file_name": r[0], "file_path": r[1],
            "source_type": r[2], "error_message": r[3],
            "failed_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]
    return {
        "failed_count": len(failed),
        "failed_documents": failed,
        "note": "Use POST /api/v1/ingest with file_path to re-ingest each document",
    }


# ── 6. Document Management ─────────────────────────────────────────────────────
@router.get("/admin/documents")
async def list_documents(
    source_type: Optional[str] = None,
    limit: int = 50,
    user: UserInToken = Depends(require_admin),
):
    """List indexed documents with chunk counts and embedding status."""
    with get_db() as conn:
        cur = conn.cursor()
        where = "WHERE source_type = %s" if source_type else ""
        params = [source_type, limit] if source_type else [limit]
        cur.execute(f"""
            SELECT title, source_type, language,
                   COUNT(*) as chunk_count,
                   MIN(created_at) as first_ingested,
                   MAX(created_at) as last_ingested
            FROM documents_bgem3
            {where}
            GROUP BY title, source_type, language
            ORDER BY last_ingested DESC NULLS LAST
            LIMIT %s
        """, params)
        rows = cur.fetchall()
        cur.close()

    return [
        {
            "title": r[0], "source_type": r[1], "language": r[2],
            "chunk_count": r[3],
            "first_ingested": r[4].isoformat() if r[4] else None,
            "last_ingested": r[5].isoformat() if r[5] else None,
            "embedding_model": "BAAI/bge-m3",
        }
        for r in rows
    ]


@router.delete("/admin/documents/{title}")
async def delete_document(
    title: str,
    user: UserInToken = Depends(require_admin),
):
    """Delete all chunks for a document by title."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents_bgem3 WHERE title = %s", [title])
        count = cur.fetchone()[0]
        if count == 0:
            raise HTTPException(status_code=404, detail=f"Document '{title}' not found")
        cur.execute("DELETE FROM documents_bgem3 WHERE title = %s", [title])
        conn.commit()
        cur.close()
    logger.info(f"Admin {user.email} deleted document '{title}' ({count} chunks)")
    return {"deleted": True, "title": title, "chunks_removed": count}


@router.post("/admin/documents/{title}/reembed")
async def reembed_document(
    title: str,
    user: UserInToken = Depends(require_admin),
):
    """
    Trigger re-embedding for a specific document.
    Returns instructions since re-embedding requires the original file.
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*), source_type,
                   metadata->>'file_path' as file_path
            FROM documents_bgem3
            WHERE title = %s
            GROUP BY source_type, metadata->>'file_path'
            LIMIT 1
        """, [title])
        row = cur.fetchone()
        cur.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Document '{title}' not found")

    return {
        "title": title,
        "current_chunks": row[0],
        "source_type": row[1],
        "file_path": row[2],
        "instructions": f"To re-embed: DELETE chunks then POST /api/v1/ingest with file_path='{row[2]}'",
    }


# ── 7. Cache Management ────────────────────────────────────────────────────────
@router.get("/admin/cache/stats")
async def get_cache_stats(user: UserInToken = Depends(require_admin)):
    """View semantic cache size, hit rate, and oldest entry."""
    try:
        import redis as redis_lib
        r = redis_lib.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

        # Semantic cache index
        index_data = r.get("glass_ai:semantic:index")
        index = json.loads(index_data) if index_data else []

        # Redis memory info
        info = r.info("memory")
        keyspace = r.info("keyspace")

        # Count all glass_ai keys
        all_keys = r.keys("glass_ai:*")
        retrieval_keys = [k for k in all_keys if "retrieval" in k]
        semantic_keys = [k for k in all_keys if "semantic" in k]

        return {
            "status": "connected",
            "semantic_cache": {
                "index_entries": len(index),
                "max_entries": 500,
                "threshold": float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92")),
                "oldest_query": index[0].get("query") if index else None,
                "newest_query": index[-1].get("query") if index else None,
            },
            "exact_cache": {
                "cached_queries": len(retrieval_keys),
            },
            "redis_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
            "total_glass_ai_keys": len(all_keys),
        }
    except Exception as e:
        return {"status": "unavailable", "error": str(e)}


@router.post("/admin/cache/flush")
async def flush_cache(user: UserInToken = Depends(require_admin)):
    """Flush all Glass Expert AI cache entries from Redis."""
    try:
        import redis as redis_lib
        r = redis_lib.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)
        keys = r.keys("glass_ai:*")
        if keys:
            r.delete(*keys)
        logger.info(f"Admin {user.email} flushed cache — {len(keys)} keys deleted")
        return {"flushed": True, "keys_deleted": len(keys)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache flush failed: {str(e)}")


# ── 8. Rate Limit Config ───────────────────────────────────────────────────────
@router.get("/admin/rate-limits")
async def get_rate_limits(user: UserInToken = Depends(require_admin)):
    """View current rate limit configuration."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT scope, requests_per_minute, updated_by, updated_at FROM rate_limit_config ORDER BY scope")
        rows = cur.fetchall()
        cur.close()

    return {
        "current_active": int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")),
        "configured_limits": [
            {
                "scope": r[0],
                "requests_per_minute": r[1],
                "updated_by": r[2],
                "updated_at": r[3].isoformat() if r[3] else None,
            }
            for r in rows
        ],
        "note": "Update RATE_LIMIT_PER_MINUTE in .env to change active limit. DB config is for audit trail.",
    }


class RateLimitUpdateRequest(BaseModel):
    requests_per_minute: int
    scope: str = "global"


@router.patch("/admin/rate-limits")
async def update_rate_limits(
    body: RateLimitUpdateRequest,
    user: UserInToken = Depends(require_admin),
):
    """Update rate limit configuration — takes effect on next request."""
    if body.requests_per_minute < 1 or body.requests_per_minute > 1000:
        raise HTTPException(status_code=400, detail="requests_per_minute must be between 1 and 1000")

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO rate_limit_config (scope, requests_per_minute, updated_by, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (scope) DO UPDATE
            SET requests_per_minute = EXCLUDED.requests_per_minute,
                updated_by = EXCLUDED.updated_by,
                updated_at = NOW()
        """, [body.scope, body.requests_per_minute, user.email])
        conn.commit()
        cur.close()

    # Update in-memory rate limit
    import api.main as main_module
    main_module._RATE_LIMIT_MAX = body.requests_per_minute

    logger.info(f"Admin {user.email} updated rate limit to {body.requests_per_minute}/min for scope '{body.scope}'")
    return {
        "updated": True,
        "scope": body.scope,
        "requests_per_minute": body.requests_per_minute,
        "effective": "immediate",
        "updated_by": user.email,
    }