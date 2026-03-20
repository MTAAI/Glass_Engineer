"""
Glass Expert AI — Admin Dashboard Router
Provides stats and metrics for admin users.
"""
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from api.auth import require_admin, UserInToken
from api.database import get_db

router = APIRouter()


@router.get("/admin/stats")
async def get_dashboard_stats(user: UserInToken = Depends(require_admin)):
    """Get system-wide statistics for the admin dashboard."""
    with get_db() as conn:
        cur = conn.cursor()
        stats = {}

        # Total chunks
        cur.execute("SELECT count(*) FROM documents_bgem3")
        stats["total_chunks"] = cur.fetchone()[0]

        # Chunks by source type
        cur.execute(
            "SELECT source_type, count(*) FROM documents_bgem3 GROUP BY source_type ORDER BY count(*) DESC"
        )
        stats["chunks_by_type"] = [
            {"source_type": row[0], "count": row[1]} for row in cur.fetchall()
        ]

        # Chunks by language
        cur.execute(
            "SELECT language, count(*) FROM documents_bgem3 GROUP BY language ORDER BY count(*) DESC"
        )
        stats["chunks_by_language"] = [
            {"language": row[0], "count": row[1]} for row in cur.fetchall()
        ]

        # Total users
        cur.execute("SELECT count(*) FROM users")
        stats["total_users"] = cur.fetchone()[0]

        # Active users (queried in last 7 days)
        cur.execute(
            """SELECT count(DISTINCT user_id) FROM chat_history
               WHERE created_at > now() - interval '7 days' AND user_id IS NOT NULL"""
        )
        stats["active_users_7d"] = cur.fetchone()[0]

        # Total conversations
        cur.execute("SELECT count(DISTINCT session_id) FROM chat_history")
        stats["total_conversations"] = cur.fetchone()[0]

        # Total messages
        cur.execute("SELECT count(*) FROM chat_history")
        stats["total_messages"] = cur.fetchone()[0]

        # Messages last 7 days
        cur.execute(
            "SELECT count(*) FROM chat_history WHERE created_at > now() - interval '7 days'"
        )
        stats["messages_7d"] = cur.fetchone()[0]

        # Feedback summary
        cur.execute(
            """SELECT
                 count(*) as total,
                 count(*) FILTER (WHERE rating = 1) as positive,
                 count(*) FILTER (WHERE rating = -1) as negative
               FROM feedback"""
        )
        row = cur.fetchone()
        stats["feedback"] = {
            "total": row[0],
            "positive": row[1],
            "negative": row[2],
            "satisfaction_pct": round(row[1] / row[0] * 100, 1) if row[0] > 0 else 0,
        }

        # Top queried topics (from chat_history, last 30 days)
        cur.execute(
            """SELECT content, count(*) as cnt
               FROM chat_history
               WHERE role = 'user' AND created_at > now() - interval '30 days'
               GROUP BY content
               ORDER BY cnt DESC
               LIMIT 10"""
        )
        stats["top_queries"] = [
            {"question": row[0][:100], "count": row[1]} for row in cur.fetchall()
        ]

        # Ingestion log (recent 20)
        cur.execute(
            """SELECT file_name, source_type, language, chunk_count, status, created_at
               FROM ingestion_log
               ORDER BY created_at DESC
               LIMIT 20"""
        )
        stats["recent_ingestions"] = [
            {
                "file_name": row[0],
                "source_type": row[1],
                "language": row[2],
                "chunk_count": row[3],
                "status": row[4],
                "created_at": row[5].isoformat() if hasattr(row[5], 'isoformat') else str(row[5]),
            }
            for row in cur.fetchall()
        ]

        # Model usage (from chat_history metadata)
        cur.execute(
            """SELECT
                 metadata->>'model_used' as model,
                 count(*) as cnt
               FROM chat_history
               WHERE role = 'assistant' AND metadata->>'model_used' IS NOT NULL
               GROUP BY metadata->>'model_used'
               ORDER BY cnt DESC"""
        )
        stats["model_usage"] = [
            {"model": row[0], "count": row[1]} for row in cur.fetchall()
        ]

        cur.close()

    return stats

@router.get("/admin/analytics")
async def get_query_analytics(user: UserInToken = Depends(require_admin)):
    """
    Query analytics — latency stats, fallback rate, slowest queries.
    Critical for production monitoring with 300 users.
    """
    with get_db() as conn:
        cur = conn.cursor()
        analytics = {}

        # Overall latency stats
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

        # Queries per hour (last 24h)
        cur.execute("""
            SELECT
                DATE_TRUNC('hour', created_at) as hour,
                COUNT(*) as queries
            FROM query_analytics
            WHERE created_at > now() - interval '24 hours'
            GROUP BY hour
            ORDER BY hour DESC
        """)
        analytics["queries_per_hour"] = [
            {"hour": row[0].isoformat(), "queries": row[1]}
            for row in cur.fetchall()
        ]

        # Fallback rate
        cur.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE fallback_used = TRUE) as fallbacks
            FROM query_analytics
            WHERE created_at > now() - interval '24 hours'
        """)
        row = cur.fetchone()
        total = row[0] or 1
        analytics["fallback_rate_pct"] = round(row[1] / total * 100, 1)
        analytics["fallback_count"] = row[1]

        # Fallback reasons breakdown
        cur.execute("""
            SELECT fallback_reason, COUNT(*) as cnt
            FROM query_analytics
            WHERE fallback_used = TRUE
            AND created_at > now() - interval '24 hours'
            GROUP BY fallback_reason
            ORDER BY cnt DESC
        """)
        analytics["fallback_reasons"] = [
            {"reason": row[0], "count": row[1]}
            for row in cur.fetchall()
        ]

        # Language breakdown
        cur.execute("""
            SELECT language, COUNT(*) as cnt
            FROM query_analytics
            WHERE created_at > now() - interval '24 hours'
            GROUP BY language
            ORDER BY cnt DESC
        """)
        analytics["queries_by_language"] = [
            {"language": row[0], "count": row[1]}
            for row in cur.fetchall()
        ]

        # Slowest queries (top 10)
        cur.execute("""
            SELECT query_text, total_latency_ms, retrieval_latency_ms,
                   model_used, created_at
            FROM query_analytics
            WHERE created_at > now() - interval '24 hours'
            ORDER BY total_latency_ms DESC
            LIMIT 10
        """)
        analytics["slowest_queries"] = [
            {
                "query": row[0][:100],
                "total_ms": round(row[1], 0),
                "retrieval_ms": round(row[2], 0),
                "model": row[3],
                "time": row[4].isoformat(),
            }
            for row in cur.fetchall()
        ]

        cur.close()
        return analytics