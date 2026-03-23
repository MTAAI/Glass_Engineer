"""
Glass Expert AI - Farsi RAG Pipeline Evaluation
=================================================
Evaluates the full RAG pipeline for Persian/Farsi queries.
"""
import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Fix Windows console encoding for Farsi/Persian output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
FARSI_SET = PROJECT_ROOT / "data" / "evaluation" / "persian_eval_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"
JUDGE_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 120

JUDGE_PROMPT = """You are an expert in glass science. Evaluate the following answer which should be in Persian/Farsi.
Question (Farsi): {question}
Reference Answer (Farsi): {reference}
Model Answer: {answer}

Evaluation criteria:
1. Technical accuracy (does it match the reference answer's content?)
2. Language (is the response in Farsi? English responses to Farsi questions score lower)
3. Completeness (does it cover the key points?)

Score 1-5: 5=excellent (accurate + Farsi), 4=good, 3=acceptable, 2=poor, 1=wrong/irrelevant.
Respond with ONLY a JSON object: {{"score": N, "reason": "one sentence", "language_ok": true/false}}"""


def _get_auth_headers(api_url: str) -> dict:
    eval_email = os.getenv("EVAL_EMAIL", "")
    eval_password = os.getenv("EVAL_PASSWORD", "")
    if not eval_email or not eval_password:
        return {}
    try:
        resp = requests.post(
            f"{api_url}/api/v1/auth/login",
            data={"username": eval_email, "password": eval_password},
            timeout=10,
        )
        if resp.ok:
            token = resp.json().get("access_token", "")
            return {"Authorization": f"Bearer {token}"}
    except Exception:
        pass
    return {}


def query_rag_api(api_url, question, headers=None):
    start = time.time()
    resp = requests.post(
        f"{api_url}/api/v1/query",
        json={"question": question, "top_k": 5},
        headers=headers or {},
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
        return int(data["score"]), data.get("reason", ""), data.get("language_ok", False)
    except Exception:
        return 1, "parse error", False


def main():
    parser = argparse.ArgumentParser(description="Evaluate Glass Expert AI Farsi RAG pipeline")
    parser.add_argument("--url", default="http://localhost:8081", help="API base URL")
    parser.add_argument("--label", default="farsi-eval", help="Label for results")
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    args = parser.parse_args()
    api_url = args.url.rstrip("/")

    if not FARSI_SET.exists():
        print(f"ERROR: Farsi eval set not found at {FARSI_SET}")
        sys.exit(1)

    questions = []
    with open(FARSI_SET, encoding="utf-8") as f:
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
    print(f"  Glass Expert AI — Farsi Evaluation ({len(questions)} questions)")
    print(f"  API: {api_url}  |  Judge: {JUDGE_MODEL}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    sys.stdout.flush()

    client = OpenAI()
    auth_headers = _get_auth_headers(api_url)
    if auth_headers:
        print(f"  Auth: authenticated")
    else:
        print(f"  Auth: anonymous (set EVAL_EMAIL/EVAL_PASSWORD for auth)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"eval_results_{args.label}_{ts}.jsonl"
    summary_file = RESULTS_DIR / f"eval_summary_{args.label}_{ts}.json"

    results = []
    total_judge = 0
    farsi_ok_count = 0
    errors = 0

    for i, q in enumerate(questions, 1):
        question = q.get("question", "")
        reference = q.get("answer", q.get("reference_answer", ""))
        category = q.get("category", "Unknown")
        difficulty = q.get("difficulty", "unknown")

        print(f"\n[{i:02d}/{len(questions)}] {category} ({difficulty})")
        print(f"  Q: {question[:80]}...")
        sys.stdout.flush()

        try:
            rag_resp, latency = query_rag_api(api_url, question, auth_headers)
            answer = rag_resp.get("answer", "")
            model_used = rag_resp.get("model_used", "unknown")
            lang_detected = rag_resp.get("language_detected", "unknown")
            sources = rag_resp.get("sources", [])
            n_sources = len(sources)

            judge_score, judge_reason, lang_ok = judge_answer(client, question, reference, answer)

            sym = "OK" if judge_score >= 4 else ("~~" if judge_score == 3 else "XX")
            lang_sym = "FA" if lang_ok else "EN!"
            print(f"  [{sym}] judge={judge_score}/5  lang={lang_sym}  detected={lang_detected}")
            print(f"  Model: {model_used}  |  Sources: {n_sources}  |  Latency: {latency:.0f}ms")
            print(f"  Reason: {judge_reason}")
            sys.stdout.flush()

            result = {
                "id": i, "category": category, "difficulty": difficulty,
                "question": question, "reference": reference,
                "answer": answer[:500],
                "judge_score": judge_score, "judge_reason": judge_reason,
                "language_ok": lang_ok, "language_detected": lang_detected,
                "latency_ms": round(latency, 0),
                "model_used": model_used, "n_sources": n_sources,
            }
            results.append(result)
            total_judge += judge_score
            if lang_ok:
                farsi_ok_count += 1

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
    farsi_pct = farsi_ok_count / n * 100

    summary = {
        "label": args.label,
        "timestamp": ts,
        "n_questions": n,
        "errors": errors,
        "mean_judge_score": round(mean_judge, 4),
        "farsi_language_accuracy_pct": round(farsi_pct, 1),
        "farsi_ok_count": farsi_ok_count,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 0),
        "models_used": dict(sorted(
            {m: sum(1 for r in results if r["model_used"] == m)
             for m in set(r["model_used"] for r in results)}.items()
        )),
        "by_category": {
            c: round(sum(r["judge_score"] for r in results if r["category"] == c) /
                     sum(1 for r in results if r["category"] == c), 2)
            for c in sorted(set(r["category"] for r in results))
        },
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"  FARSI EVALUATION COMPLETE — {args.label}")
    print("=" * 70)
    print(f"  Questions answered  : {n} / {n + errors}")
    print(f"  Errors              : {errors}")
    print(f"  LLM Judge Score     : {mean_judge:.2f} / 5.00")
    print(f"  Farsi Language OK   : {farsi_ok_count}/{n} ({farsi_pct:.0f}%)")
    print(f"  Avg latency         : {sum(r['latency_ms'] for r in results) / n:.0f}ms")
    print(f"  Models used         : {summary['models_used']}")
    print(f"  Results saved to    : {results_file}")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
