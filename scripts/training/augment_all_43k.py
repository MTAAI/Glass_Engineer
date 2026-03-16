"""
Glass Expert AI — Full 43K Answer Upgrade (10 concurrent workers)
================================================================
Upgrades ALL Q&A pairs to expert quality using GPT-4o-mini with
asyncio concurrency for 10x speed.

Cost estimate: ~$15 for 43K pairs
Time estimate: ~1.5 hours with 10 workers

Usage:
    python scripts/training/augment_all_43k.py                    # full run
    python scripts/training/augment_all_43k.py --limit=100        # test run
    python scripts/training/augment_all_43k.py --workers=20       # more speed (risk rate limit)
    python scripts/training/augment_all_43k.py --skip-good        # skip answers already >200 words

Output:
    data/processed/augmented_all_43k.jsonl  — all upgraded answers
"""

import json
import os
import sys
import asyncio
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Add repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "augmented_all_43k.jsonl"
MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
SEMAPHORE_LIMIT = 50  # concurrent API calls
RPM_LIMIT = 2000      # requests per minute (GPT-4o-mini tier 2+ = 2000-10000 RPM)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("augment_all_43k.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────

UPGRADE_SYSTEM = """You are a glass science expert with PhD-level knowledge rewriting Q&A pairs for a training dataset.

Your job: take ANY glass science Q&A pair and produce the BEST possible expert answer.

Rules:
1. Include specific numerical values (temperatures in °C and K, compositions in mol% or wt%, properties with SI units)
2. Reference glass science principles: Zachariasen rules, VFT equation, Adam-Gibbs theory, network theory (BO/NBO), Griffith theory, etc.
3. Cover 3-5 key technical points
4. 150-400 words — detailed but focused
5. Use well-known textbook values (e.g., soda-lime Tg ~570°C, borosilicate CTE ~3.3×10⁻⁶/°C, E-glass tensile strength ~3.4 GPa)
6. Structure: direct answer first → detailed explanation → specific examples/values
7. Do NOT fabricate obscure values — use well-established data
8. If the original answer is already excellent (detailed, specific, well-structured), keep it mostly intact but ensure completeness

Return JSON: {"answer": "your expert answer"}"""


# ── Async GPT Client ─────────────────────────────────────────────────────────

async def call_gpt(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    rate_limiter: "RateLimiter",
    question: str,
    original_answer: str,
    word_count: int,
) -> dict | None:
    """Call GPT-4o-mini to upgrade one answer."""
    async with semaphore:
        await rate_limiter.acquire()

        user_prompt = f"""Question: {question}

Current answer ({word_count} words): {original_answer}

Rewrite into a detailed, expert-level glass science answer with specific values and principles."""

        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": UPGRADE_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.5,
                    max_tokens=1500,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                data = json.loads(content)
                new_answer = data.get("answer", "")
                if new_answer and len(new_answer.split()) >= 20:
                    return {
                        "question": question,
                        "answer": new_answer,
                        "original_answer": original_answer,
                        "original_word_count": word_count,
                        "new_word_count": len(new_answer.split()),
                        "source": "upgraded_all",
                    }
                else:
                    logger.warning(f"Short answer for: {question[:60]}...")
                    return None
            except json.JSONDecodeError as e:
                logger.warning(f"JSON error (attempt {attempt+1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                err_str = str(e)
                if "rate_limit" in err_str.lower() or "429" in err_str:
                    wait = 2 ** (attempt + 2)  # 4, 8, 16 seconds
                    logger.warning(f"Rate limit hit, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    logger.warning(f"API error (attempt {attempt+1}): {e}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2)
        return None


class RateLimiter:
    """Token-bucket rate limiter for RPM compliance."""
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.interval = 60.0 / rpm  # seconds between requests
        self.last_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            wait = self.interval - (now - self.last_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_time = time.monotonic()


# ── Load & Save ──────────────────────────────────────────────────────────────

def load_all_pairs() -> list[dict]:
    """Load all Q&A pairs from all sources."""
    import glob

    pairs = []
    seen_questions = set()

    # Load from all JSONL files
    patterns = [
        str(REPO_ROOT / "data" / "processed" / "*.jsonl"),
        str(REPO_ROOT / "data" / "qa_pairs" / "**" / "*.jsonl"),
    ]

    skip_files = {"augmented_upgraded.jsonl", "augmented_expanded.jsonl",
                  "augmented_all_43k.jsonl", "master_qa_v3.jsonl"}

    for pattern in patterns:
        for fpath in glob.glob(pattern, recursive=True):
            if Path(fpath).name in skip_files:
                continue
            try:
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            q = d.get("question", "").strip()
                            a = d.get("answer", "").strip()
                            if q and a and q not in seen_questions:
                                seen_questions.add(q)
                                pairs.append({"question": q, "answer": a})
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                logger.warning(f"Error reading {fpath}: {e}")

    return pairs


def load_checkpoint() -> set[str]:
    """Load already-processed questions from checkpoint."""
    done = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    done.add(d.get("question", ""))
                except:
                    pass
    return done


def save_result(result: dict, file_handle):
    """Save one result to the output file."""
    file_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    file_handle.flush()


# ── Main Pipeline ────────────────────────────────────────────────────────────

async def process_batch(
    client: AsyncOpenAI,
    pairs: list[dict],
    file_handle,
    workers: int,
    stats: dict,
):
    """Process all pairs with concurrent workers."""
    semaphore = asyncio.Semaphore(workers)
    rate_limiter = RateLimiter(RPM_LIMIT)

    total = len(pairs)
    start_time = time.time()

    async def process_one(idx: int, pair: dict):
        q = pair["question"]
        a = pair["answer"]
        wc = len(a.split())

        result = await call_gpt(client, semaphore, rate_limiter, q, a, wc)

        if result:
            save_result(result, file_handle)
            stats["upgraded"] += 1
        else:
            # Keep original if upgrade fails
            fallback = {
                "question": q,
                "answer": a,
                "original_word_count": wc,
                "new_word_count": wc,
                "source": "original_kept",
            }
            save_result(fallback, file_handle)
            stats["kept_original"] += 1

        stats["processed"] += 1

        if stats["processed"] % 100 == 0:
            elapsed = time.time() - start_time
            rate = stats["processed"] / elapsed * 60
            eta_min = (total - stats["processed"]) / max(rate, 1)
            logger.info(
                f"  [{stats['processed']:,}/{total:,}] "
                f"upgraded={stats['upgraded']:,} kept={stats['kept_original']:,} "
                f"rate={rate:.0f}/min ETA={eta_min:.0f}min"
            )

    # Create all tasks
    tasks = [process_one(i, p) for i, p in enumerate(pairs)]

    # Run with progress
    await asyncio.gather(*tasks)


async def main_async(args):
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in .env")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    # Load all pairs
    all_pairs = load_all_pairs()
    logger.info(f"Loaded {len(all_pairs):,} unique Q&A pairs")

    # Filter by skip-good flag
    if args.skip_good:
        before = len(all_pairs)
        all_pairs = [p for p in all_pairs if len(p["answer"].split()) < 200]
        logger.info(f"Filtered to {len(all_pairs):,} pairs (skipped {before - len(all_pairs):,} with >200 word answers)")

    # Resume from checkpoint
    done = load_checkpoint()
    if done:
        logger.info(f"Checkpoint: {len(done):,} already processed")
        all_pairs = [p for p in all_pairs if p["question"] not in done]
        logger.info(f"Remaining: {len(all_pairs):,} to process")

    # Apply limit
    if args.limit > 0:
        all_pairs = all_pairs[:args.limit]
        logger.info(f"Limited to {len(all_pairs):,} pairs")

    if not all_pairs:
        logger.info("Nothing to process — all done!")
        return

    # Sort shortest first (weakest answers get processed first)
    all_pairs.sort(key=lambda p: len(p["answer"].split()))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info(f"Glass Expert AI — Full Answer Upgrade")
    logger.info(f"  Pairs to process: {len(all_pairs):,}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Model: {MODEL}")
    logger.info(f"  Output: {OUTPUT_FILE}")
    logger.info("=" * 65)

    stats = {"processed": 0, "upgraded": 0, "kept_original": 0}
    start = time.time()

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fh:
        await process_batch(client, all_pairs, fh, args.workers, stats)

    elapsed = time.time() - start
    logger.info("=" * 65)
    logger.info("UPGRADE COMPLETE")
    logger.info(f"  Processed:      {stats['processed']:,}")
    logger.info(f"  Upgraded:       {stats['upgraded']:,}")
    logger.info(f"  Kept original:  {stats['kept_original']:,}")
    logger.info(f"  Time:           {elapsed/60:.1f} min")
    logger.info(f"  Rate:           {stats['processed']/elapsed*60:.0f}/min")
    logger.info(f"  Output:         {OUTPUT_FILE}")
    logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Upgrade ALL 43K+ Q&A pairs with GPT-4o-mini")
    parser.add_argument("--limit", type=int, default=0, help="Limit pairs (0=all)")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent API workers")
    parser.add_argument("--skip-good", action="store_true", help="Skip answers already >200 words")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
