"""
Glass Expert AI — Retrieval-Only Evaluation (no OpenAI key needed)
Runs all 50 golden questions and scores retrieval quality.
"""
import json, time, requests, sys
from pathlib import Path

BASE_URL    = "http://localhost:8080"
GOLDEN_SET  = Path("data/evaluation/golden_eval_set.jsonl")

def login():
    r = requests.post(f"{BASE_URL}/api/v1/auth/login",
                      data={"username": "arjun@glassai.com", "password": "glass2024"})
    return r.json()["access_token"]

def query(token, question):
    start = time.time()
    r = requests.post(f"{BASE_URL}/api/v1/query",
                      headers={"Authorization": f"Bearer {token}"},
                      json={"question": question, "top_k": 5})
    return r.json(), (time.time() - start) * 1000

def score(result, reference):
    if not result.get("sources"):
        return 0, "no sources"
    top      = result["sources"][0]
    rerank   = top.get("rerank_score") or 0
    sim      = top.get("similarity") or 0
    n_src    = len(result["sources"])
    # Keyword coverage
    ref_words = set(w.lower() for w in reference.split() if len(w) > 4)
    ans_words = set(w.lower() for w in result.get("answer","").split() if len(w) > 4)
    kw_cov    = len(ref_words & ans_words) / len(ref_words) if ref_words else 0
    s = 0
    if rerank > 0.8:  s += 2
    elif rerank > 0.4: s += 1
    if sim > 0.5:     s += 1
    if n_src >= 3:    s += 1
    if kw_cov > 0.1:  s += 1
    return s, f"rerank={rerank:.3f} sim={sim:.3f} kw={kw_cov:.0%} src={n_src}"

def main():
    questions = []
    with open(GOLDEN_SET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print("=" * 65)
    print(f"  Glass Expert AI — Retrieval Evaluation ({len(questions)} questions)")
    print("=" * 65)

    token = login()
    print("✅ Logged in\n")

    total, results = 0, []
    for i, q in enumerate(questions, 1):
        question  = q.get("question", "")
        reference = q.get("answer", q.get("reference_answer", ""))
        category  = q.get("category", "General")
        print(f"[{i:02d}/{len(questions)}] {category} — {question[:55]}...")
        try:
            result, ms = query(token, question)
            s, note    = score(result, reference)
            total += s
            sym = "✅" if s >= 4 else ("⚠️" if s >= 2 else "❌")
            print(f"         {sym} {s}/5 | {note} | {ms:.0f}ms")
            results.append({"question": question, "category": category,
                             "score": s, "note": note, "latency_ms": ms})
        except Exception as e:
            print(f"         ❌ Error: {e}")
            results.append({"question": question, "category": category,
                             "score": 0, "note": str(e), "latency_ms": 0})

    n      = len(questions)
    avg    = total / n
    pct    = (total / (n * 5)) * 100
    good   = sum(1 for r in results if r["score"] >= 4)
    ok     = sum(1 for r in results if r["score"] == 3)
    poor   = sum(1 for r in results if r["score"] <= 2)
    avg_ms = sum(r["latency_ms"] for r in results) / n

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["score"])

    print("\n" + "=" * 65)
    print("RESULTS")
    print(f"  Total score  : {total} / {n * 5}")
    print(f"  Average      : {avg:.2f} / 5.00")
    print(f"  Composite    : {pct:.1f}%")
    print(f"  Good  (4-5)  : {good}")
    print(f"  OK    (3)    : {ok}")
    print(f"  Poor  (0-2)  : {poor}")
    print(f"  Avg latency  : {avg_ms:.0f}ms")
    print("\nBy Category:")
    for cat, scores in sorted(by_cat.items()):
        print(f"  {cat:<30} {sum(scores)/len(scores):.2f}/5")
    print("=" * 65)

    out = {"total": total, "max": n*5, "average": round(avg,4),
           "composite_pct": round(pct,2), "by_category": {
               c: round(sum(v)/len(v),2) for c,v in by_cat.items()},
           "results": results}
    with open("golden_retrieval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("✅ Saved to golden_retrieval_results.json")

if __name__ == "__main__":
    main()