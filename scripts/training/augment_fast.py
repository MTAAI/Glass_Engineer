"""
Glass Expert AI — FAST Parallel Data Augmentation
===================================================
Async parallel workers for GPT-4o-mini augmentation.
10-20x faster than sequential version.

Resumes from existing checkpoint (augmented_upgraded.jsonl).

Usage:
    python scripts/training/augment_fast.py --workers=15 --limit=5000
    python scripts/training/augment_fast.py --workers=20 --limit=5000  # aggressive
"""
import json
import os
import sys
import asyncio
import argparse
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
MASTER_QA = OUTPUT_DIR / "master_qa.jsonl"
CHECKPOINT = OUTPUT_DIR / "augmented_upgraded.jsonl"

MODEL = "gpt-4o-mini"
MAX_RETRIES = 3

UPGRADE_SYSTEM = """You are a glass science expert rewriting Q&A pairs for a training dataset.
Your job: take a short or vague answer and rewrite it into a detailed, precise, expert-level answer.

Rules:
- Include specific numerical values (temperatures in °C, compositions in mol%, properties with units)
- Reference glass science principles (network theory, Zachariasen rules, VFT equation, etc.)
- Cover at least 3 key points in the answer
- Keep the answer between 150-400 words
- Use technical but clear language
- Do NOT fabricate values — use well-known textbook values when possible

Return JSON: {"upgraded_answer": "your detailed answer"}"""


# ── Thread-safe file writer ──────────────────────────────────────────────────
class IncrementalWriter:
    """Thread-safe incremental JSONL writer."""
    def __init__(self, path: Path):
        self.path = path
        self.lock = asyncio.Lock()
        self.count = 0

    async def write(self, entry: dict):
        async with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.count += 1


# ── Async GPT call ──────────────────────────────────────────────────────────
async def call_gpt_async(client, question: str, answer: str, word_count: int, sem: asyncio.Semaphore) -> dict | None:
    """Call GPT-4o-mini with semaphore-based concurrency control."""
    async with sem:
        user_prompt = f"""Question: {question}

Current answer ({word_count} words): {answer}

Rewrite this into a detailed, expert-level answer with specific values and glass science principles."""

        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": UPGRADE_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.5,
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content)
                new_answer = data.get("upgraded_answer", "")
                if new_answer and len(new_answer.split()) > word_count:
                    return {
                        "question": question,
                        "answer": new_answer,
                        "source": "upgraded",
                        "original_word_count": word_count,
                    }
                return None
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 2 ** (attempt + 1) + 1
                    await asyncio.sleep(wait)
                else:
                    await asyncio.sleep(1)
        return None


async def process_batch(targets: list[tuple], client, writer: IncrementalWriter, sem: asyncio.Semaphore, total: int):
    """Process a batch of targets concurrently."""
    tasks = []
    for pair_idx, word_count, question, answer in targets:
        task = asyncio.create_task(
            call_gpt_async(client, question, answer, word_count, sem)
        )
        tasks.append(task)

    start = time.time()
    completed = 0
    errors = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            await writer.write(result)
            completed += 1
        else:
            errors += 1

        done = completed + errors
        if done % 50 == 0 or done == len(tasks):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            total_done = writer.count
            remaining = total - total_done
            eta_min = remaining / rate / 60 if rate > 0 else 0
            print(f"  [{total_done}/{total}] +{completed} ok, {errors} err | {rate:.1f}/sec | ETA: {eta_min:.0f} min")

    return completed, errors


async def main_async(args):
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing data
    if not MASTER_QA.exists():
        print(f"ERROR: Master Q&A not found: {MASTER_QA}")
        sys.exit(1)

    pairs = []
    with open(MASTER_QA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    print(f"{'=' * 65}")
    print(f"  Glass Expert AI — FAST Parallel Augmentation")
    print(f"  Pairs: {len(pairs):,}  |  Workers: {args.workers}  |  Model: {MODEL}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 65}")

    # Sort by answer length (shortest = weakest)
    scored = []
    for i, p in enumerate(pairs):
        wc = len(p["answer"].split())
        scored.append((i, wc, p["question"], p["answer"]))
    scored.sort(key=lambda x: x[1])

    # Take the shortest answers
    limit = args.limit
    targets = scored[:limit]
    print(f"  Targeting {len(targets)} shortest answers")

    # Resume: skip already-processed
    done_questions = set()
    if CHECKPOINT.exists():
        with open(CHECKPOINT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        done_questions.add(entry["question"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"  Checkpoint: {len(done_questions)} already done")

    targets = [(i, wc, q, a) for i, wc, q, a in targets if q not in done_questions]
    remaining = len(targets)
    total = limit  # total target count
    print(f"  Remaining: {remaining} to process")

    if remaining == 0:
        print("  Nothing to do — all targets already upgraded!")
        return

    # Semaphore controls max concurrent API calls
    sem = asyncio.Semaphore(args.workers)
    writer = IncrementalWriter(CHECKPOINT)
    writer.count = len(done_questions)  # set starting count

    # Process in batches of 200 to show progress
    batch_size = 200
    total_ok = 0
    total_err = 0

    for batch_start in range(0, len(targets), batch_size):
        batch = targets[batch_start:batch_start + batch_size]
        ok, err = await process_batch(batch, client, writer, sem, total)
        total_ok += ok
        total_err += err

    print(f"\n{'=' * 65}")
    print(f"  PHASE 1 COMPLETE")
    print(f"  Upgraded: {total_ok}  |  Errors: {total_err}")
    print(f"  Total in checkpoint: {writer.count}")
    print(f"  File: {CHECKPOINT}")
    print(f"{'=' * 65}")


def main():
    parser = argparse.ArgumentParser(description="Fast parallel data augmentation")
    parser.add_argument("--workers", type=int, default=15,
                        help="Number of concurrent API calls (default: 15)")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Total pairs to upgrade (default: 5000)")
    args = parser.parse_args()

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
