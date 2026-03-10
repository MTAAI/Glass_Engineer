"""
Glass Expert AI - RAG Pipeline Evaluation
==========================================
Evaluates the full RAG pipeline (retrieval + LLM) against the 50-question golden set.
Queries through the /api/v1/query endpoint and compares with the fine-tuned LLM-only baseline.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
GOLDEN_SET = PROJECT_ROOT / "data" / "evaluation" / "golden_eval_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"
JUDGE_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 120

JUDGE_PROMPT = """You are an expert in glass science and engineering. Evaluate the following answer.
Question: {question}
Reference Answer: {reference}
Model Answer: {answer}
Score 1-5: 5=excellent, 4=good, 3=acceptable, 2=poor, 1=wrong.
Respond with ONLY a JSON object: {{"score": N, "reason": "one sentence"}}"""


def query_rag_api(api_url, question):
    start = time.time()
    resp = requests.post(
        f"{api_url}/api/v1/query",
        json={"question": question, "top_k": 5},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    latency = (time.time() - start) * 1000
    return data, latency


def judge_answer(client, question, reference, answer):
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content.strip())
        return int(data["score"]), data.get("reason", "")
    except Exception:
        return 1, "parse error"


def keyword_coverage(reference, answer):
    strip_chars = ".,;:()"
    ref = set(w.lower().strip(strip_chars) for w in reference.split() if len(w) > 3)
    ans = set(w.lower().strip(strip_chars) for w in answer.split() if len(w) > 3)
    return len(ref & ans) / len(ref) if ref else 1.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Glass Expert AI RAG pipeline")
    parser.add_argument("--url", default="http://localhost:8081", help="API base URL")
    parser.add_argument("--label", default="rag-pipeline", help="Label for results")
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    args = parser.parse_args()
    api_url = args.url.rstrip("/")

    if not GOLDEN_SET.exists():
        print(f"ERROR: Golden set not found at {GOLDEN_SET}")
        sys.exit(1)

    questions = []
    with open(GOLDEN_SET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if args.limit > 0:
        questions = questions[:args.limit]

    print("=" * 70)
    print(f"  Glass Expert AI — RAG Pipeline Evaluation ({len(questions)} questions)")
    print(f"  API: {api_url}  |  Judge: {JUDGE_MODEL}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    sys.stdout.flush()

    client = OpenAI()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"eval_results_{args.label}_{ts}.jsonl"
    summary_file = RESULTS_DIR / f"eval_summary_{args.label}_{ts}.json"

    results = []
    total_judge = 0
    total_kw = 0
    total_retrieval_ms = 0
    errors = 0
    similarities = []

    for i, q in enumerate(questions, 1):
        question = q.get("question", "")
        reference = q.get("answer", q.get("reference_answer", ""))
        category = q.get("category", "Unknown")
        difficulty = q.get("difficulty", "unknown")

        print(f"\n[{i:02d}/{len(questions)}] {category} ({difficulty})")
        print(f"  Q: {question[:80]}...")
        sys.stdout.flush()

        try:
            rag_resp, latency = query_rag_api(api_url, question)
            answer = rag_resp.get("answer", "")
            model_used = rag_resp.get("model_used", "unknown")
            retrieval_ms = rag_resp.get("retrieval_time_ms", 0)
            sources = rag_resp.get("sources", [])
            n_sources = len(sources)
            avg_sim = sum(s.get("similarity", 0) for s in sources) / n_sources if n_sources else 0
            top_sim = max((s.get("similarity", 0) for s in sources), default=0)

            total_retrieval_ms += retrieval_ms
            similarities.append(avg_sim)

            judge_score, judge_reason = judge_answer(client, question, reference, answer)
            kw_score = keyword_coverage(reference, answer)
            composite = (judge_score / 5.0 * 0.7 + kw_score * 0.3) * 100

            sym = "OK" if judge_score >= 4 else ("~~" if judge_score == 3 else "XX")
            print(f"  [{sym}] judge={judge_score}/5  kw={kw_score:.0%}  composite={composite:.1f}%")
            print(f"  Model: {model_used}  |  Sources: {n_sources}  |  Top sim: {top_sim:.3f}  |  Latency: {latency:.0f}ms")
            print(f"  Reason: {judge_reason}")
            sys.stdout.flush()

            result = {
                "id": i, "category": category, "difficulty": difficulty,
                "question": question, "reference": reference, "answer": answer,
                "judge_score": judge_score, "judge_reason": judge_reason,
                "kw_score": round(kw_score, 4), "composite": round(composite, 2),
                "latency_ms": round(latency, 0),
                "model_used": model_used,
                "n_sources": n_sources, "avg_similarity": round(avg_sim, 4),
                "top_similarity": round(top_sim, 4),
                "retrieval_ms": round(retrieval_ms, 2),
            }
            results.append(result)
            total_judge += judge_score
            total_kw += kw_score

            with open(results_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"  ERROR: {e}")
            sys.stdout.flush()
            errors += 1

    n = len(results)
    if n == 0:
        print("\nNo results to summarize!")
        return

    mean_judge = total_judge / n
    mean_kw = total_kw / n
    composite = (mean_judge / 5.0 * 0.7 + mean_kw * 0.3) * 100
    score_dist = {str(s): sum(1 for r in results if r["judge_score"] == s) for s in range(1, 6)}
    mean_sim = sum(similarities) / len(similarities) if similarities else 0

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["judge_score"])

    summary = {
        "label": args.label,
        "timestamp": ts,
        "n_questions": n,
        "errors": errors,
        "mean_judge_score": round(mean_judge, 4),
        "mean_kw_coverage": round(mean_kw, 4),
        "composite_score_pct": round(composite, 2),
        "score_distribution": score_dist,
        "by_category": {c: round(sum(v) / len(v), 2) for c, v in sorted(by_cat.items())},
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 0),
        "avg_retrieval_ms": round(total_retrieval_ms / n, 2),
        "avg_similarity": round(mean_sim, 4),
        "models_used": dict(sorted(
            {m: sum(1 for r in results if r["model_used"] == m) for m in set(r["model_used"] for r in results)}.items()
        )),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Comparison with baseline ──────────────────────────────────────────────
    baseline_file = RESULTS_DIR / "eval_summary_v2-improved-local_20260306_105210.json"
    baseline = None
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)

    print()
    print("=" * 70)
    print(f"  RAG PIPELINE EVALUATION COMPLETE — {args.label}")
    print("=" * 70)
    print(f"  Questions answered : {n} / {n + errors}")
    print(f"  Errors             : {errors}")
    print(f"  LLM Judge Score    : {mean_judge:.2f} / 5.00", end="")
    if baseline:
        diff = mean_judge - baseline["mean_judge_score"]
        print(f"  (baseline: {baseline['mean_judge_score']:.2f}, {'+'if diff>=0 else ''}{diff:.2f})")
    else:
        print()
    print(f"  Keyword Coverage   : {mean_kw:.1%}", end="")
    if baseline:
        diff = mean_kw - baseline["mean_kw_coverage"]
        print(f"  (baseline: {baseline['mean_kw_coverage']:.1%}, {'+'if diff>=0 else ''}{diff:.1%})")
    else:
        print()
    print(f"  Composite Score    : {composite:.1f}%", end="")
    if baseline:
        diff = composite - baseline["composite_score_pct"]
        print(f"  (baseline: {baseline['composite_score_pct']:.1f}%, {'+'if diff>=0 else ''}{diff:.1f}%)")
    else:
        print()
    print(f"  Score distribution : {score_dist}")
    print(f"  Avg retrieval      : {total_retrieval_ms / n:.0f}ms")
    print(f"  Avg similarity     : {mean_sim:.3f}")
    print(f"  Avg total latency  : {sum(r['latency_ms'] for r in results) / n:.0f}ms")
    print(f"  Models used        : {summary['models_used']}")
    print(f"  Results saved to   : {results_file}")
    print(f"  Summary saved to   : {summary_file}")
    print("=" * 70)

    if baseline:
        print()
        print("  CATEGORY COMPARISON (RAG vs Fine-tuned LLM baseline):")
        print("  " + "-" * 55)
        all_cats = sorted(set(list(summary["by_category"].keys()) + list(baseline.get("by_category", {}).keys())))
        for cat in all_cats:
            rag_score = summary["by_category"].get(cat, 0)
            base_score = baseline.get("by_category", {}).get(cat, 0)
            diff = rag_score - base_score if base_score else 0
            arrow = "^" if diff > 0 else ("v" if diff < 0 else "=")
            print(f"  {cat:<30s} RAG: {rag_score:.2f}  Base: {base_score:.2f}  [{arrow}{abs(diff):+.2f}]")
        print("  " + "-" * 55)

    sys.stdout.flush()


if __name__ == "__main__":
    main()
