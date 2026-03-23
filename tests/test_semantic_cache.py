"""
Glass Expert AI — Semantic Cache Hit Rate Evaluation
=====================================================
Runs questions through the semantic cache to measure:
  - Exact cache hit rate (same question twice)
  - Semantic cache hit rate (paraphrased questions)
  - Latency improvement from caching

Run with:
    python tests/test_semantic_cache.py
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
    if GOLDEN_SET.exists():
        with open(GOLDEN_SET, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        q = json.loads(line)
                        questions.append(q.get("question", ""))
                    except Exception:
                        pass
    if not questions:
        # Fallback test questions
        questions = [
            "What is the glass transition temperature of borosilicate glass?",
            "How does viscosity change with temperature in glass melts?",
            "What causes devitrification in glass?",
            "What is the coefficient of thermal expansion of soda-lime glass?",
            "How is float glass manufactured?",
        ]
    return [q for q in questions if q]


def run_cache_eval():
    questions = load_questions()
    n = min(len(questions), 20)  # test first 20
    questions = questions[:n]

    print(f"\n{'='*60}")
    print(f"Semantic Cache Hit Rate Evaluation — {n} questions")
    print(f"{'='*60}\n")

    # Pass 1: Populate cache (cold)
    print("Pass 1: Populating cache...")
    cold_times = []
    for i, q in enumerate(questions, 1):
        start = time.time()
        retrieve_with_auto_language(q, top_k=5)
        elapsed = (time.time() - start) * 1000
        cold_times.append(elapsed)
        if i % 5 == 0:
            print(f"  {i}/{n} cached...")

    avg_cold = sum(cold_times) / len(cold_times)
    print(f"\nPass 1 done. Avg cold retrieval: {avg_cold:.0f}ms\n")

    # Pass 2: Exact cache hits
    print("Pass 2: Re-running same questions (expect exact cache hits)...")
    cache_hits = 0
    warm_times = []
    for q in questions:
        start = time.time()
        retrieve_with_auto_language(q, top_k=5)
        elapsed = (time.time() - start) * 1000
        warm_times.append(elapsed)
        if elapsed < 500:
            cache_hits += 1

    avg_warm = sum(warm_times) / len(warm_times)
    hit_rate = cache_hits / n * 100

    print(f"\n{'='*60}")
    print(f"  CACHE RESULTS")
    print(f"{'='*60}")
    print(f"  Total questions     : {n}")
    print(f"  Exact cache hits    : {cache_hits}")
    print(f"  Hit rate            : {hit_rate:.1f}%")
    print(f"  Avg cold latency    : {avg_cold:.0f}ms")
    print(f"  Avg warm latency    : {avg_warm:.0f}ms")
    if avg_warm > 0:
        print(f"  Speedup             : {avg_cold/avg_warm:.1f}x")
    print(f"{'='*60}\n")

    # Pass 3: Paraphrase test (semantic cache)
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
        start = time.time()
        retrieve_with_auto_language(paraphrase, top_k=5)
        elapsed = (time.time() - start) * 1000
        hit = elapsed < 500
        if hit:
            semantic_hits += 1
        status = "HIT" if hit else "MISS"
        print(f"  {status} ({elapsed:.0f}ms) '{paraphrase[:50]}'")

    print(f"\n  Paraphrase hits: {semantic_hits}/{len(paraphrases)}")
    print(f"  Semantic threshold: 0.92 cosine")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_cache_eval()
