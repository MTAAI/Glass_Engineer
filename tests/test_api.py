"""
Glass Expert AI — API Test Suite
Tests the full RAG pipeline via the FastAPI endpoints.
Run with: python tests/test_api.py
"""
import sys
import httpx
from loguru import logger

API_BASE  = "http://localhost:8080/api/v1"
TEST_USER = "arjun@glassai.com"
TEST_PASS = "glass2024"

logger.remove()
logger.add(sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO")

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def get_auth_headers(client: httpx.Client) -> dict:
    r = client.post(f"{API_BASE}/auth/login",
                    data={"username": TEST_USER, "password": TEST_PASS})
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_health(client: httpx.Client):
    print_header("TEST 1: Health Check")
    r    = client.get(f"{API_BASE}/health", timeout=10)
    data = r.json()
    print(f"  Status:          {data['status']}")
    print(f"  Database:        {data['database']}")
    print(f"  Redis:           {data['redis']}")
    print(f"  Embedding model: {data['embedding_model']}")
    print(f"  Total documents: {data['total_documents']}")
    print(f"  Total chunks:    {data['total_chunks']}")
    assert data["database"] == "healthy", "Database is not healthy!"
    logger.info("✅ Health check passed")

def test_query_english(client: httpx.Client, headers: dict):
    print_header("TEST 2: English Query")
    questions = [
        "What is the glass transition temperature of borosilicate glass?",
        "What causes devitrification in glass manufacturing?",
        "How does viscosity change with temperature in glass melts?",
    ]
    for question in questions:
        print(f"\n  Query: {question}")
        r    = client.post(f"{API_BASE}/query",
                           headers=headers,
                           json={"question": question, "top_k": 3},
                           timeout=120)
        data = r.json()
        print(f"  Language detected: {data.get('language_detected', 'N/A')}")
        print(f"  Sources found:     {len(data.get('sources', []))}")
        print(f"  Retrieval time:    {data.get('retrieval_time_ms', 0):.1f}ms")
        print(f"  Model used:        {data.get('model_used', 'N/A')}")
        print(f"  Answer preview:    {data.get('answer', '')[:200]}...")
        if data.get("sources"):
            top = data["sources"][0]
            print(f"  Top source:        {top['title']} (sim={top['similarity']:.4f})")
    logger.info("✅ English query test passed")

def test_query_farsi(client: httpx.Client, headers: dict):
    print_header("TEST 3: Farsi Query")
    question = "دمای انتقال شیشه چیست؟"
    print(f"\n  Query: {question}")
    r    = client.post(f"{API_BASE}/query",
                       headers=headers,
                       json={"question": question, "top_k": 3},
                       timeout=120)
    data = r.json()
    print(f"  Language detected: {data.get('language_detected', 'N/A')}")
    print(f"  Sources found:     {len(data.get('sources', []))}")
    print(f"  Answer preview:    {data.get('answer', '')[:200]}...")
    logger.info("✅ Farsi query test passed")

def test_query_with_filter(client: httpx.Client, headers: dict):
    print_header("TEST 4: Query with Source Type Filter")
    r    = client.post(f"{API_BASE}/query",
                       headers=headers,
                       json={"question": "What are the corrective actions for glass defects?",
                             "top_k": 5, "source_type": "sop"},
                       timeout=120)
    data = r.json()
    print(f"  Filter:        source_type=sop")
    print(f"  Sources found: {len(data.get('sources', []))}")
    for s in data.get("sources", []):
        print(f"    - {s['title']} [{s['source_type']}] sim={s['similarity']:.4f}")
    logger.info("✅ Filtered query test passed")

def test_no_results_graceful(client: httpx.Client, headers: dict):
    print_header("TEST 5: Graceful No-Results Handling")
    r    = client.post(f"{API_BASE}/query",
                       headers=headers,
                       json={"question": "What is the population of Tokyo?", "top_k": 3},
                       timeout=120)
    data = r.json()
    print(f"  Sources found: {len(data.get('sources', []))}")
    print(f"  Answer:        {data.get('answer', '')[:200]}")
    logger.info("✅ Graceful no-results test passed")

def test_conversations(client: httpx.Client, headers: dict):
    print_header("TEST 6: Conversations")

    # List existing conversations
    r = client.get(f"{API_BASE}/conversations", headers=headers, timeout=10)
    convs = r.json().get("conversations", r.json() if isinstance(r.json(), list) else [])
    print(f"  Conversations listed: {len(convs)}")

    # Create a new session and send a query to populate chat_history
    r = client.post(f"{API_BASE}/conversations", headers=headers,
                    json={"title": "Test Conversation"}, timeout=10)
    conv    = r.json()
    sid     = conv.get("session_id")
    print(f"  Created session_id: {sid}")
    assert sid, "No session_id returned"

    # Send a query with this session_id so chat_history gets populated
    client.post(f"{API_BASE}/query", headers=headers,
                json={"question": "What is glass?", "top_k": 1,
                      "session_id": sid}, timeout=120)
    print(f"  Query sent to populate session")

    # Now rename should work
    r = client.patch(f"{API_BASE}/conversations/{sid}", headers=headers,
                     json={"title": "Renamed Test"}, timeout=10)
    print(f"  Rename status: {r.status_code}")
    assert r.status_code == 200, f"Rename failed: {r.text}"

    # Delete should work
    r = client.delete(f"{API_BASE}/conversations/{sid}", headers=headers, timeout=10)
    print(f"  Delete status: {r.status_code}")
    assert r.status_code == 200, f"Delete failed: {r.text}"

    logger.info("✅ Conversations test passed")

if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("Glass Expert AI — API Test Suite")
    print("🔬 " * 20)

    try:
        with httpx.Client() as client:
            headers = get_auth_headers(client)
            logger.info(f"✅ Authenticated as {TEST_USER}")

            test_health(client)
            test_query_english(client, headers)
            test_query_farsi(client, headers)
            test_query_with_filter(client, headers)
            test_no_results_graceful(client, headers)
            test_conversations(client, headers)

        print("\n" + "✅ " * 20)
        print("All API tests passed!")
        print("✅ " * 20)

    except httpx.ConnectError:
        logger.error("❌ Cannot connect to API. Start the server first:")
        logger.error("   python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload")
        sys.exit(1)
    except AssertionError as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)