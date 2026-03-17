"""
Glass Expert AI — Retrieval-Only Evaluation
============================================
Tests RAG retrieval quality (vector search + keyword + reranking) WITHOUT LLM.
Measures: similarity scores, keyword hit rate, source relevance, latency.
Runs against both English and Farsi eval sets.

Usage:
    python scripts/evaluation/eval_retrieval_only.py
    python scripts/evaluation/eval_retrieval_only.py --lang en --limit 10
    python scripts/evaluation/eval_retrieval_only.py --lang fa
    python scripts/evaluation/eval_retrieval_only.py --lang both
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────
EN_EVAL_SET = PROJECT_ROOT / "data" / "evaluation" / "golden_eval_set.jsonl"
FA_EVAL_SET = PROJECT_ROOT / "data" / "evaluation" / "persian_eval_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"
TOP_K = 5


def load_eval_set(path: Path) -> list:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return questions


def keyword_overlap(reference: str, chunks: list) -> float:
    """What fraction of reference answer keywords appear in retrieved chunks."""
    strip = ".,;:()[]{}\"'"
    ref_words = set(
        w.lower().strip(strip)
        for w in reference.split()
        if len(w) > 3 and not w.startswith("[")
    )
    if not ref_words:
        return 1.0

    all_content = " ".join(c.get("content", "") for c in chunks).lower()
    hits = sum(1 for w in ref_words if w in all_content)
    return hits / len(ref_words)


def check_keyword_hits(question_keywords: list, chunks: list) -> dict:
    """Check how many question-level keywords are found in retrieved chunks."""
    if not question_keywords:
        return {"total": 0, "found": 0, "pct": 0.0}
    all_content = " ".join(c.get("content", "") + " " + c.get("title", "") for c in chunks).lower()
    found = sum(1 for kw in question_keywords if kw.lower() in all_content)
    return {
        "total": len(question_keywords),
        "found": found,
        "pct": round(found / len(question_keywords) * 100, 1),
    }


def run_eval(questions: list, lang_label: str, top_k: int = TOP_K) -> dict:
    """Run retrieval evaluation on a set of questions."""
    from retrieval.retriever import retrieve_with_auto_language

    results = []
    total_sim = 0
    total_top_sim = 0
    total_kw_overlap = 0
    total_kw_hit_pct = 0
    total_latency = 0
    empty_retrievals = 0

    print(f"\n{'='*70}")
    print(f"  Retrieval Evaluation — {lang_label.upper()} ({len(questions)} questions)")
    print(f"  top_k={top_k}  |  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    sys.stdout.flush()

    for i, q in enumerate(questions, 1):
        question = q.get("question", "")
        reference = q.get("answer", q.get("reference_answer", ""))
        category = q.get("category", "Unknown")
        difficulty = q.get("difficulty", "unknown")
        keywords = q.get("keywords", [])

        start = time.time()
        try:
            chunks, detected_lang = retrieve_with_auto_language(
                query=question,
                top_k=top_k,
            )
            latency_ms = (time.time() - start) * 1000
        except Exception as e:
            print(f"  [{i:02d}] ERROR: {e}")
            sys.stdout.flush()
            results.append({"id": i, "error": str(e)})
            continue

        n_chunks = len(chunks)
        if n_chunks == 0:
            empty_retrievals += 1

        # Metrics
        sims = [c.get("similarity", 0) for c in chunks]
        rerank_scores = [c.get("rerank_score", 0) for c in chunks if "rerank_score" in c]
        avg_sim = sum(sims) / len(sims) if sims else 0
        top_sim = max(sims) if sims else 0
        avg_rerank = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0
        top_rerank = max(rerank_scores) if rerank_scores else 0

        kw_overlap = keyword_overlap(reference, chunks)
        kw_hits = check_keyword_hits(keywords, chunks)

        total_sim += avg_sim
        total_top_sim += top_sim
        total_kw_overlap += kw_overlap
        total_kw_hit_pct += kw_hits["pct"]
        total_latency += latency_ms

        # Source type distribution
        source_types = {}
        for c in chunks:
            st = c.get("source_type", "unknown")
            source_types[st] = source_types.get(st, 0) + 1

        # Quality rating
        if top_sim >= 0.7 and kw_overlap >= 0.4:
            quality = "GOOD"
        elif top_sim >= 0.5 and kw_overlap >= 0.2:
            quality = "OK  "
        elif n_chunks > 0:
            quality = "WEAK"
        else:
            quality = "MISS"

        sym = {"GOOD": "+", "OK  ": "~", "WEAK": "!", "MISS": "X"}[quality]

        print(f"\n  [{i:02d}/{len(questions)}] {sym} {quality} | {category} ({difficulty})")
        print(f"    Q: {question[:75]}{'...' if len(question) > 75 else ''}")
        print(f"    Chunks: {n_chunks} | Top sim: {top_sim:.3f} | Avg sim: {avg_sim:.3f}", end="")
        if rerank_scores:
            print(f" | Top rerank: {top_rerank:.3f}", end="")
        print()
        print(f"    Keyword overlap: {kw_overlap:.0%} | Domain kw hits: {kw_hits['found']}/{kw_hits['total']} ({kw_hits['pct']:.0f}%)")
        print(f"    Latency: {latency_ms:.0f}ms | Lang detected: {detected_lang} | Sources: {source_types}")

        # Show top 2 chunk titles for verification
        for j, c in enumerate(chunks[:2], 1):
            title = c.get("title", "?")[:60]
            sim = c.get("similarity", 0)
            rs = c.get("rerank_score", "n/a")
            print(f"    → [{j}] sim={sim:.3f} rr={rs} '{title}'")

        sys.stdout.flush()

        result = {
            "id": i,
            "category": category,
            "difficulty": difficulty,
            "question": question[:200],
            "language": lang_label,
            "detected_language": detected_lang,
            "n_chunks": n_chunks,
            "avg_similarity": round(avg_sim, 4),
            "top_similarity": round(top_sim, 4),
            "avg_rerank_score": round(avg_rerank, 4),
            "top_rerank_score": round(top_rerank, 4),
            "keyword_overlap": round(kw_overlap, 4),
            "domain_kw_hits": kw_hits,
            "latency_ms": round(latency_ms, 0),
            "source_types": source_types,
            "quality": quality.strip(),
        }
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len([r for r in results if "error" not in r])
    errors = len([r for r in results if "error" in r])

    if n == 0:
        print("\n  No results to summarize!")
        return {"error": "no results"}

    mean_sim = total_sim / n
    mean_top_sim = total_top_sim / n
    mean_kw_overlap = total_kw_overlap / n
    mean_kw_hit_pct = total_kw_hit_pct / n
    mean_latency = total_latency / n

    quality_dist = {}
    for r in results:
        if "error" not in r:
            q = r.get("quality", "?")
            quality_dist[q] = quality_dist.get(q, 0) + 1

    # By category
    by_cat = {}
    for r in results:
        if "error" not in r:
            cat = r["category"]
            by_cat.setdefault(cat, []).append(r["top_similarity"])

    print(f"\n{'='*70}")
    print(f"  RETRIEVAL EVALUATION SUMMARY — {lang_label.upper()}")
    print(f"{'='*70}")
    print(f"  Questions evaluated : {n} / {n + errors}")
    print(f"  Errors              : {errors}")
    print(f"  Empty retrievals    : {empty_retrievals}")
    print(f"  Mean top similarity : {mean_top_sim:.4f}")
    print(f"  Mean avg similarity : {mean_sim:.4f}")
    print(f"  Mean keyword overlap: {mean_kw_overlap:.1%}")
    print(f"  Mean domain kw hits : {mean_kw_hit_pct:.1f}%")
    print(f"  Mean latency        : {mean_latency:.0f}ms")
    print(f"  Quality distribution: {quality_dist}")
    print(f"\n  By category (avg top similarity):")
    for cat, sims_list in sorted(by_cat.items()):
        avg = sum(sims_list) / len(sims_list)
        print(f"    {cat:<30s}: {avg:.4f}  (n={len(sims_list)})")
    print(f"{'='*70}")
    sys.stdout.flush()

    summary = {
        "label": lang_label,
        "timestamp": datetime.now().isoformat(),
        "n_questions": n,
        "errors": errors,
        "empty_retrievals": empty_retrievals,
        "mean_top_similarity": round(mean_top_sim, 4),
        "mean_avg_similarity": round(mean_sim, 4),
        "mean_keyword_overlap": round(mean_kw_overlap, 4),
        "mean_domain_kw_hit_pct": round(mean_kw_hit_pct, 2),
        "mean_latency_ms": round(mean_latency, 0),
        "quality_distribution": quality_dist,
        "by_category": {c: round(sum(v)/len(v), 4) for c, v in sorted(by_cat.items())},
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality (no LLM)")
    parser.add_argument("--lang", default="both", choices=["en", "fa", "both"])
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Retrieval top_k")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_summaries = {}

    if args.lang in ("en", "both"):
        if not EN_EVAL_SET.exists():
            print(f"ERROR: English eval set not found at {EN_EVAL_SET}")
        else:
            questions = load_eval_set(EN_EVAL_SET)
            if args.limit > 0:
                questions = questions[:args.limit]
            summary = run_eval(questions, "en", args.top_k)
            all_summaries["en"] = summary

    if args.lang in ("fa", "both"):
        if not FA_EVAL_SET.exists():
            print(f"ERROR: Farsi eval set not found at {FA_EVAL_SET}")
        else:
            questions = load_eval_set(FA_EVAL_SET)
            if args.limit > 0:
                questions = questions[:args.limit]
            summary = run_eval(questions, "fa", args.top_k)
            all_summaries["fa"] = summary

    # Save combined results
    output_file = RESULTS_DIR / f"retrieval_eval_{args.lang}_{ts}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # Final comparison if both
    if "en" in all_summaries and "fa" in all_summaries:
        en = all_summaries["en"]
        fa = all_summaries["fa"]
        if "error" not in en and "error" not in fa:
            print(f"\n{'='*70}")
            print(f"  EN vs FA COMPARISON")
            print(f"{'='*70}")
            print(f"  {'Metric':<25s} {'English':>10s} {'Farsi':>10s}")
            print(f"  {'-'*45}")
            print(f"  {'Top similarity':<25s} {en['mean_top_similarity']:>10.4f} {fa['mean_top_similarity']:>10.4f}")
            print(f"  {'Avg similarity':<25s} {en['mean_avg_similarity']:>10.4f} {fa['mean_avg_similarity']:>10.4f}")
            print(f"  {'Keyword overlap':<25s} {en['mean_keyword_overlap']:>10.1%} {fa['mean_keyword_overlap']:>10.1%}")
            print(f"  {'Domain kw hits':<25s} {en['mean_domain_kw_hit_pct']:>9.1f}% {fa['mean_domain_kw_hit_pct']:>9.1f}%")
            print(f"  {'Latency (ms)':<25s} {en['mean_latency_ms']:>10.0f} {fa['mean_latency_ms']:>10.0f}")
            print(f"  {'Empty retrievals':<25s} {en['empty_retrievals']:>10d} {fa['empty_retrievals']:>10d}")
            print(f"  {'-'*45}")
            print(f"  {'Quality: GOOD':<25s} {en['quality_distribution'].get('GOOD',0):>10d} {fa['quality_distribution'].get('GOOD',0):>10d}")
            print(f"  {'Quality: OK':<25s} {en['quality_distribution'].get('OK',0):>10d} {fa['quality_distribution'].get('OK',0):>10d}")
            print(f"  {'Quality: WEAK':<25s} {en['quality_distribution'].get('WEAK',0):>10d} {fa['quality_distribution'].get('WEAK',0):>10d}")
            print(f"  {'Quality: MISS':<25s} {en['quality_distribution'].get('MISS',0):>10d} {fa['quality_distribution'].get('MISS',0):>10d}")
            print(f"{'='*70}")

    print(f"\n  Results saved to: {output_file}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
