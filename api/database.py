"""
Glass Expert AI — Database Connection Pool
Centralized connection pooling using psycopg2.pool.
All routers should use `get_db()` instead of raw `psycopg2.connect()`.
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

# Thread-safe pool: 2 connections idle, up to 20 under load
_pool: psycopg2.pool.ThreadedConnectionPool | None = None


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
        logger.info("DB connection pool initialized (min=2, max=20)")
    except Exception as e:
        logger.error(f"Failed to initialize DB pool: {e}")
        raise


def _shutdown_pool():
    global _pool
    if _pool:
        _pool.closeall()
        logger.info("DB connection pool closed")


atexit.register(_shutdown_pool)


@contextmanager
def get_db():
    """Get a pooled database connection. Auto-returns to pool on exit.

    Usage:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(...)
            conn.commit()  # if writing
    """
    if _pool is None:
        _init_pool()

    conn = _pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def get_db_conn():
    """Get a raw pooled connection (for code that manages its own lifecycle).
    MUST call return_db_conn() when done."""
    if _pool is None:
        _init_pool()
    return _pool.getconn()


def return_db_conn(conn):
    """Return a connection to the pool."""
    if _pool:
        _pool.putconn(conn)
