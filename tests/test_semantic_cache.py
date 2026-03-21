"""
Glass Expert AI — Semantic Cache Hit Rate Evaluation
=====================================================
Runs all 50 golden set questions through the semantic cache.
Reports:
  - How many hit cache (paraphrases)
  - Avg cache hit latency vs full retrieval latency
  - Overall hit rate %
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.retriever import retrieve_with_auto_language

GOLDEN_SET = Path("data/evaluation/golden_eval_set.jsonl")

def load_questions():
    questions = []
    with open(GOLDEN_SET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    q = json.loads(line)
                    questions.append(q.get("question", ""))
                except Exception:
                    pass
    return [q for q in questions if q]

def run_cache_eval():
    questions = load_questions()
    print(f"\n{'='*60}")
    print(f"Semantic Cache Hit Rate Evaluation — {len(questions)} questions")
    print(f"{'='*60}\n")

    # First pass — populate cache (no cache hits expected)
    print("Pass 1: Populating cache with all 50 questions...")
    cold_times = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        retrieve_with_auto_language(q, top_k=5)
        elapsed = (time.time() - start) * 1000
        cold_times.append(elapsed)
        if i % 10 == 0:
            print(f"  {i}/50 questions cached...")

    avg_cold = sum(cold_times) / len(cold_times)
    print(f"\nPass 1 complete. Avg cold retrieval: {avg_cold:.0f}ms\n")

    # Second pass — should get cache hits for exact matches
    print("Pass 2: Re-running same 50 questions (expect exact cache hits)...")
    cache_hits = 0
    warm_times = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        retrieve_with_auto_language(q, top_k=5)
        elapsed = (time.time() - start) * 1000
        warm_times.append(elapsed)
        # Cache hit if much faster than cold
        if elapsed < 500:
            cache_hits += 1

    avg_warm = sum(warm_times) / len(warm_times)
    hit_rate = cache_hits / len(questions) * 100

    print(f"\nPass 2 complete.")
    print(f"\n{'='*60}")
    print(f"  SEMANTIC CACHE RESULTS")
    print(f"{'='*60}")
    print(f"  Total questions     : {len(questions)}")
    print(f"  Cache hits          : {cache_hits}")
    print(f"  Hit rate            : {hit_rate:.1f}%")
    print(f"  Avg cold latency    : {avg_cold:.0f}ms")
    print(f"  Avg warm latency    : {avg_warm:.0f}ms")
    print(f"  Latency improvement : {avg_cold/avg_warm:.1f}x faster")
    print(f"{'='*60}\n")

    # Third pass — paraphrase test
    paraphrases = [
        ("What is the glass transition temperature?", "What is Tg in glass?"),
        ("How is float glass made?", "Explain the float glass manufacturing process"),
        ("What causes devitrification?", "Why does glass crystallize?"),
        ("What is the refractive index of glass?", "How is light refracted in glass?"),
        ("What are glass defects?", "What types of defects occur in glass manufacturing?"),
    ]

    print("Pass 3: Paraphrase semantic cache test...")
    semantic_hits = 0
    for original, paraphrase in paraphrases:
        # Original should be cached from pass 1
        start = time.time()
        retrieve_with_auto_language(paraphrase, top_k=5)
        elapsed = (time.time() - start) * 1000
        hit = elapsed < 500
        if hit:
            semantic_hits += 1
        status = "HIT ✅" if hit else "MISS ❌"
        print(f"  {status} ({elapsed:.0f}ms) '{paraphrase[:50]}'")

    print(f"\n  Paraphrase hits: {semantic_hits}/{len(paraphrases)}")
    print(f"  Semantic cache threshold: 0.92 cosine similarity")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_cache_eval()