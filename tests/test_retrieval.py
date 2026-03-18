"""
Glass Expert AI — Retrieval Test Suite
Tests the full pipeline: embedding generation, vector search, and result quality.

Run with:
    python tests/test_retrieval.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from retrieval.retriever import (
    retrieve,
    retrieve_with_auto_language,
    format_context_for_llm,
    detect_language,
)


# ── Test Queries ───────────────────────────────────────────────────────────────
ENGLISH_QUERIES = [
    "What is the glass transition temperature?",
    "How does viscosity change with temperature in glass melts?",
    "What causes devitrification in glass?",
    "Explain the Vogel-Fulcher-Tammann equation for glass viscosity.",
    "What is the effect of Na2O on the thermal expansion coefficient of glass?",
    "How do you calculate the refractive index of a glass composition?",
    "What are the main types of glass defects in manufacturing?",
    "What is the annealing point of soda-lime glass?",
]

FARSI_QUERIES = [
    "دمای انتقال شیشه چیست؟",                    # What is the glass transition temperature?
    "ویسکوزیته شیشه مذاب چگونه تغییر می‌کند؟",    # How does viscosity of glass melt change?
    "دیویتریفیکاسیون در شیشه چیست؟",              # What is devitrification in glass?
]


# ── Test Functions ─────────────────────────────────────────────────────────────
def test_database_connection():
    """Test that the database is accessible and has documents."""
    import os
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv()

    logger.info("=" * 60)
    logger.info("TEST 1: Database Connection")
    logger.info("=" * 60)

    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM documents_bgem3;")
        doc_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT title) FROM documents_bgem3;")
        title_count = cur.fetchone()[0]

        cur.execute("SELECT source_type, COUNT(*) FROM documents_bgem3 GROUP BY source_type;")
        breakdown = cur.fetchall()

        cur.close()
        conn.close()

        logger.info(f"✅ Database connected successfully")
        logger.info(f"   Total chunks: {doc_count:,}")
        logger.info(f"   Unique documents: {title_count}")
        logger.info(f"   Breakdown by type:")
        for row in breakdown:
            logger.info(f"     {row[0]}: {row[1]:,} chunks")

        if doc_count == 0:
            logger.warning("⚠️  No documents found! Run the ingestion pipeline first:")
            logger.warning("   python ingestion/ingest.py data/sample_docs/ --type textbook")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error("   Make sure Docker is running: cd docker && docker compose up -d")
        return False


def test_language_detection():
    """Test automatic language detection."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Language Detection")
    logger.info("=" * 60)

    test_cases = [
        ("What is the glass transition temperature?", "en"),
        ("دمای انتقال شیشه چیست؟", "fa"),
        ("How does viscosity change?", "en"),
        ("ویسکوزیته شیشه مذاب", "fa"),
    ]

    all_passed = True
    for text, expected in test_cases:
        detected = detect_language(text)
        status = "✅" if detected == expected else "❌"
        logger.info(f"  {status} '{text[:50]}' → detected: {detected} (expected: {expected})")
        if detected != expected:
            all_passed = False

    return all_passed


def test_english_retrieval():
    """Test retrieval quality for English queries."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: English Query Retrieval")
    logger.info("=" * 60)

    for query in ENGLISH_QUERIES[:3]:  # Test first 3 queries
        logger.info(f"\n🔍 Query: {query}")
        logger.info("-" * 50)

        start = time.time()
        results = retrieve(query, top_k=3)
        elapsed = time.time() - start

        if not results:
            logger.warning(f"  ⚠️  No results found (threshold may be too high)")
            continue

        logger.info(f"  Found {len(results)} results in {elapsed:.2f}s")
        for i, r in enumerate(results, 1):
            logger.info(f"\n  Result {i}:")
            logger.info(f"    Source:     {r['title']} [{r['language'].upper()}]")
            logger.info(f"    Type:       {r['source_type']}")
            logger.info(f"    Similarity: {r['similarity']:.4f}")
            logger.info(f"    Content:    {r['content'][:150]}...")


def test_farsi_retrieval():
    """Test retrieval quality for Farsi queries."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Farsi Query Retrieval")
    logger.info("=" * 60)

    for query in FARSI_QUERIES:
        logger.info(f"\n🔍 Query: {query}")
        logger.info("-" * 50)

        start = time.time()
        results, lang = retrieve_with_auto_language(query, top_k=3)
        elapsed = time.time() - start

        logger.info(f"  Detected language: {lang.upper()}")

        if not results:
            logger.warning(f"  ⚠️  No results found")
            continue

        logger.info(f"  Found {len(results)} results in {elapsed:.2f}s")
        for i, r in enumerate(results, 1):
            logger.info(f"\n  Result {i}:")
            logger.info(f"    Source:     {r['title']} [{r['language'].upper()}]")
            logger.info(f"    Similarity: {r['similarity']:.4f}")
            logger.info(f"    Content:    {r['content'][:150]}...")


def test_context_formatting():
    """Test that retrieved results format correctly for LLM prompt."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Context Formatting for LLM")
    logger.info("=" * 60)

    query = "What is the glass transition temperature?"
    results = retrieve(query, top_k=3)

    if not results:
        logger.warning("⚠️  No results to format — skipping test")
        return

    context = format_context_for_llm(results)
    logger.info("Formatted context block:")
    logger.info("\n" + context)


def test_retrieval_speed():
    """Benchmark retrieval speed."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Retrieval Speed Benchmark")
    logger.info("=" * 60)

    query = "What is the glass transition temperature?"
    runs = 5
    times = []

    # First run (cold — no cache)
    retrieve(query, top_k=5, use_cache=False)

    for i in range(runs):
        start = time.time()
        retrieve(query, top_k=5, use_cache=False)
        elapsed = time.time() - start
        times.append(elapsed)

    avg = sum(times) / len(times)
    logger.info(f"  Runs: {runs}")
    logger.info(f"  Average retrieval time: {avg:.3f}s")
    logger.info(f"  Min: {min(times):.3f}s | Max: {max(times):.3f}s")

    # Test with cache
    start = time.time()
    retrieve(query, top_k=5, use_cache=True)
    cache_time = time.time() - start
    logger.info(f"  With Redis cache: {cache_time:.3f}s")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("\n" + "🔬 " * 20)
    logger.info("Glass Expert AI — Retrieval Test Suite")
    logger.info("🔬 " * 20 + "\n")

    # Run all tests
    db_ok = test_database_connection()

    if not db_ok:
        logger.error("\n❌ Database test failed. Fix the connection before running other tests.")
        sys.exit(1)

    test_language_detection()
    test_english_retrieval()
    test_farsi_retrieval()
    test_context_formatting()
    test_retrieval_speed()

    logger.info("\n" + "✅ " * 20)
    logger.info("All tests complete!")
    logger.info("✅ " * 20)
