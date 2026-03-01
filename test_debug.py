"""Debug script to check retrieval similarity scores directly."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')

# Lower the threshold temporarily to see all scores
import retrieval.retriever as ret
ret.SIM_THRESHOLD = 0.0  # Show all results regardless of score

from retrieval.retriever import retrieve

print("Testing retrieval with threshold=0.0 (show all scores)...")
results = retrieve('glass transition temperature borosilicate', top_k=10, use_cache=False)
print(f"\nResults found: {len(results)}")
for r in results:
    title = r.get("title", "unknown")
    score = r.get("similarity", 0)
    print(f"  Score: {score:.4f} | {title}")

if not results:
    print("  NO RESULTS AT ALL - database may be empty or embedding dimension mismatch")
    # Check DB directly
    import psycopg2
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    count = cur.fetchone()[0]
    cur.execute("SELECT id, title, array_length(embedding::text::text[], 1) FROM documents LIMIT 3")
    rows = cur.fetchall()
    print(f"\n  DB chunk count: {count}")
    for row in rows:
        print(f"  Row: id={row[0]}, title={row[1]}")
    cur.close()
    conn.close()
