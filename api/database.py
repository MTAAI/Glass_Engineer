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


def get_db_conn():
    """Get a raw pooled connection with pgvector registered.
    MUST call return_db_conn() when done.

    Raises:
        psycopg2.pool.PoolError: If all connections are exhausted.
        psycopg2.OperationalError: If DB is unreachable.
    """
    if _pool is None:
        _init_pool()
    try:
        conn = _pool.getconn()
    except psycopg2.pool.PoolError as e:
        logger.error(f"DB pool exhausted — all 20 connections in use: {e}")
        raise
    except psycopg2.OperationalError as e:
        logger.error(f"DB unreachable: {e}")
        raise

    # Auto-register pgvector on first use of each connection
    if id(conn) not in _pgvector_registered:
        try:
            from pgvector.psycopg2 import register_vector
            register_vector(conn)
            _pgvector_registered.add(id(conn))
        except ImportError:
            logger.warning("pgvector not installed — vector search unavailable")
        except psycopg2.Error as e:
            logger.warning(f"pgvector registration failed: {e}")
    return conn


def return_db_conn(conn):
    """Return a connection to the pool."""
    if _pool:
        _pool.putconn(conn)


@contextmanager
def get_db():
    """Get a pooled database connection. Auto-returns to pool on exit.
    Auto-rollback on error.

    Usage:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(...)
            conn.commit()  # if writing
    """
    conn = get_db_conn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        return_db_conn(conn)
