"""
Glass Expert AI — Database Connection Pool
Uses psycopg2 ThreadedConnectionPool — compatible with existing codebase.
All routers use get_db() context manager instead of raw psycopg2.connect().
"""
import os
import atexit
from contextlib import contextmanager

import psycopg2
import psycopg2.pool
from loguru import logger

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai",
)

_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pgvector_registered: set = set()


def _init_pool():
    global _pool
    if _pool is not None:
        return
    try:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=20,
            dsn=_DATABASE_URL,
        )
        logger.info("DB connection pool initialised (min=2, max=20)")
    except Exception as e:
        logger.error(f"Failed to initialise DB pool: {e}")
        raise


def _shutdown_pool():
    global _pool
    if _pool:
        _pool.closeall()
        logger.info("DB connection pool closed")


atexit.register(_shutdown_pool)


def get_db_conn():
    """Get a raw pooled connection. MUST call return_db_conn() when done."""
    if _pool is None:
        _init_pool()
    conn = _pool.getconn()
    # Register pgvector once per connection
    if id(conn) not in _pgvector_registered:
        try:
            from pgvector.psycopg2 import register_vector
            register_vector(conn)
            _pgvector_registered.add(id(conn))
        except Exception:
            pass
    return conn


def return_db_conn(conn):
    """Return connection to pool."""
    if _pool:
        _pool.putconn(conn)


@contextmanager
def get_db():
    """
    Context manager for pooled DB connection with auto-rollback on error.

    Usage:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(...)
            conn.commit()
    """
    if _pool is None:
        _init_pool()
    conn = get_db_conn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        return_db_conn(conn)


def check_db_health() -> dict:
    """Returns health status and document counts."""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM documents")
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL"
            )
            embedded = cur.fetchone()[0]
            cur.close()
        return {
            "status": "healthy",
            "total_documents": total,
            "total_chunks": embedded,
        }
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}