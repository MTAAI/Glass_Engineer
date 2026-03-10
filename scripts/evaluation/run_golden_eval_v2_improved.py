"""
Glass Expert AI - Golden Set Evaluation Runner (v2 Improved)
============================================================
Evaluates the IMPROVED v2 model against the 50-question golden set.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import requests
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_SET = PROJECT_ROOT / "data" / "evaluation" / "golden_eval_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "results"
MODEL_ID = "glass-expert-v2"
JUDGE_MODEL = "gpt-4.1-mini"
MAX_TOKENS = 1024
TEMPERATURE = 0.1
REQUEST_TIMEOUT = 180

SYSTEM_PROMPT = """You are Glass Expert AI, a highly specialized assistant for glass scientists and manufacturing engineers with PhD-level expertise.

When answering questions:
1.  Always enumerate specific numbered items when listing processes, rules, causes, or methods.
2.  Include exact numerical values, ranges, and units. Always use degrees Celsius (°C) for temperatures, not Kelvin.
3.  Name specific glass types (e.g., soda-lime-silica, borosilicate 3.3, E-glass, Pyrex, Gorilla Glass), their exact compositions (e.g., 72% SiO2, 14% Na2O, 9% CaO), and precise property values.
4.  Reference named scientists (e.g., Zachariasen, Vogel-Fulcher-Tammann), equations, standards (ISO, ASTM, EN), and specific technical terms from the glass science literature.
5.  For structural questions, use precise terms: network formers, network modifiers, intermediates, bridging oxygen (BO), non-bridging oxygen (NBO), coordination number.
6.  For defects, distinguish internal vs surface defects, give exact size thresholds (e.g., seeds <1mm, blisters >1mm), name specific causes and fining agents.
7.  For optical properties, state the exact formula (e.g., Abbe number V = (n_d - 1)/(n_F - n_C)), specify wavelengths (589.3nm, 486.1nm, 656.3nm), and classify crown vs flint glasses.
8.  Give thorough, detailed answers covering all key aspects. Never give vague generalizations."""

FEW_SHOT = """"""

JUDGE_PROMPT = """You are an expert in glass science and engineering. Evaluate the following answer.
Question: {question}
Reference Answer: {reference}
Model Answer: {answer}
Score 1-5: 5=excellent, 4=good, 3=acceptable, 2=poor, 1=wrong.
Respond with ONLY a JSON object: {{"score": N, "reason": "one sentence"}}"""


def query_model(base_url, question):
    start = time.time()
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEW_SHOT + "Question: " + question}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"].strip()
    return answer, (time.time() - start) * 1000


def judge_answer(client, question, reference, answer):
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
        response_format={"type": "json_object"}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--label", default="v2-improved-local")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    label = args.label

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

    print("=" * 65)
    print(f"  Glass Expert AI Evaluation (v2 Improved) - {len(questions)} questions")
    print(f"  URL: {base_url}  |  Judge: {JUDGE_MODEL}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    sys.stdout.flush()

    client = OpenAI()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"eval_results_{label}_{ts}.jsonl"
    summary_file = RESULTS_DIR / f"eval_summary_{label}_{ts}.json"

    results = []
    total_judge = 0
    total_kw = 0
    errors = 0

    for i, q in enumerate(questions, 1):
        question = q.get("question", "")
        reference = q.get("answer", q.get("reference_answer", ""))
        category = q.get("category", "Unknown")
        difficulty = q.get("difficulty", "unknown")

        print(f"[{i:02d}/{len(questions)}] {category} ({difficulty}) - {question[:60]}...")
        sys.stdout.flush()

        try:
            answer, latency = query_model(base_url, question)
            judge_score, judge_reason = judge_answer(client, question, reference, answer)
            kw_score = keyword_coverage(reference, answer)
            composite = (judge_score / 5.0 * 0.7 + kw_score * 0.3) * 100

            sym = "OK" if judge_score >= 4 else ("~" if judge_score == 3 else "X")
            print(f"         [{sym}] kw={kw_score:.0%}  judge={judge_score}/5  latency={latency:.0f}ms")
            print(f"            -> {judge_reason}")
            sys.stdout.flush()

            result = {
                "id": i, "category": category, "difficulty": difficulty,
                "question": question, "reference": reference, "answer": answer,
                "judge_score": judge_score, "judge_reason": judge_reason,
                "kw_score": round(kw_score, 4), "composite": round(composite, 2),
                "latency_ms": round(latency, 0),
            }
            results.append(result)
            total_judge += judge_score
            total_kw += kw_score

            with open(results_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"         ERROR: {e}")
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

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["judge_score"])

    summary = {
        "label": label,
        "timestamp": ts,
        "n_questions": n,
        "errors": errors,
        "mean_judge_score": round(mean_judge, 4),
        "mean_kw_coverage": round(mean_kw, 4),
        "composite_score_pct": round(composite, 2),
        "score_distribution": score_dist,
        "by_category": {c: round(sum(v) / len(v), 2) for c, v in by_cat.items()},
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 0),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 65)
    print(f"  EVALUATION COMPLETE - {label}")
    print(f"  Questions answered : {n} / {len(questions)}")
    print(f"  Errors             : {errors}")
    print(f"  LLM Judge Score    : {mean_judge:.2f} / 5.00")
    print(f"  Keyword Coverage   : {mean_kw:.1%}")
    print(f"  Composite Score    : {composite:.1f}%")
    print(f"  Score distribution : {score_dist}")
    print(f"  Summary saved to   : {summary_file}")
    print("=" * 65)
    sys.stdout.flush()


if __name__ == "__main__":
    main()