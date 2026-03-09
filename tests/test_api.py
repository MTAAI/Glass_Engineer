"""
Glass Expert AI — API Test Suite
Tests the full RAG pipeline via the FastAPI endpoints.
Run with: python tests/test_api.py
"""
import sys
import json
import time
import httpx
from loguru import logger

API_BASE = "http://localhost:8080/api/v1"

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_health():
    print_header("TEST 1: Health Check")
    r = httpx.get(f"{API_BASE}/health", timeout=10)
    data = r.json()
    print(f"  Status:          {data['status']}")
    print(f"  Database:        {data['database']}")
    print(f"  Redis:           {data['redis']}")
    print(f"  Embedding model: {data['embedding_model']}")
    print(f"  Total documents: {data['total_documents']}")
    print(f"  Total chunks:    {data['total_chunks']}")
    assert data["database"] == "healthy", "Database is not healthy!"
    logger.info("✅ Health check passed")
    return data


def test_query_english():
    print_header("TEST 2: English Query")
    questions = [
        "What is the glass transition temperature of borosilicate glass?",
        "What causes devitrification in glass manufacturing?",
        "How does viscosity change with temperature in glass melts?",
    ]
    for question in questions:
        print(f"\n  Query: {question}")
        r = httpx.post(
            f"{API_BASE}/query",
            json={"question": question, "top_k": 3},
            timeout=60,
        )
        data = r.json()
        print(f"  Language detected: {data['language_detected']}")
        print(f"  Sources found:     {len(data['sources'])}")
        print(f"  Retrieval time:    {data['retrieval_time_ms']:.1f}ms")
        print(f"  Model used:        {data['model_used']}")
        print(f"  Answer preview:    {data['answer'][:200]}...")
        if data["sources"]:
            print(f"  Top source:        {data['sources'][0]['title']} (similarity: {data['sources'][0]['similarity']:.4f})")
    logger.info("✅ English query test passed")


def test_query_farsi():
    print_header("TEST 3: Farsi Query")
    question = "دمای انتقال شیشه چیست؟"
    print(f"\n  Query: {question}")
    r = httpx.post(
        f"{API_BASE}/query",
        json={"question": question, "top_k": 3},
        timeout=60,
    )
    data = r.json()
    print(f"  Language detected: {data['language_detected']}")
    print(f"  Sources found:     {len(data['sources'])}")
    print(f"  Answer preview:    {data['answer'][:200]}...")
    logger.info("✅ Farsi query test passed")


def test_query_with_filter():
    print_header("TEST 4: Query with Source Type Filter")
    r = httpx.post(
        f"{API_BASE}/query",
        json={
            "question": "What are the corrective actions for glass defects?",
            "top_k": 5,
            "source_type": "sop",
        },
        timeout=60,
    )
    data = r.json()
    print(f"  Filter: source_type=sop")
    print(f"  Sources found: {len(data['sources'])}")
    for s in data["sources"]:
        print(f"    - {s['title']} [{s['source_type']}] sim={s['similarity']:.4f}")
    logger.info("✅ Filtered query test passed")


def test_no_results_graceful():
    print_header("TEST 5: Graceful No-Results Handling")
    r = httpx.post(
        f"{API_BASE}/query",
        json={"question": "What is the population of Tokyo?", "top_k": 3},
        timeout=60,
    )
    data = r.json()
    print(f"  Sources found: {len(data['sources'])}")
    print(f"  Answer: {data['answer'][:200]}")
    assert len(data["sources"]) == 0 or data["answer"] != "", "Should handle no results gracefully"
    logger.info("✅ Graceful no-results test passed")


if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("Glass Expert AI — API Test Suite")
    print("🔬 " * 20)

    try:
        health_data = test_health()
        test_query_english()
        test_query_farsi()
        test_query_with_filter()
        test_no_results_graceful()

        print("\n" + "✅ " * 20)
        print("All API tests complete!")
        print("✅ " * 20)

    except httpx.ConnectError:
        logger.error("❌ Cannot connect to API. Make sure the server is running:")
        logger.error("   Run: python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload")
        sys.exit(1)
    except AssertionError as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)
