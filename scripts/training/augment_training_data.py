"""
Glass Expert AI — Smart Data Augmentation
===========================================
Two-phase augmentation using GPT-4o-mini:

Phase 1: QUALITY UPGRADE (~$15-20)
  - Takes existing Q&A pairs with SHORT or WEAK answers
  - Rewrites them into detailed, RAG-grounded expert answers
  - Targets the bottom 5,000 answers by length/quality

Phase 2: TARGETED EXPANSION (~$30-50)
  - Generates new Q&A pairs for WEAK categories
  - Creates variations, follow-ups, and multi-hop questions
  - Focuses on: thermal properties, composition, melting, glass ceramics
  - Batches 3 Q&As per API call to save cost

Usage:
    python scripts/training/augment_training_data.py --phase=1          # quality upgrade
    python scripts/training/augment_training_data.py --phase=2          # targeted expansion
    python scripts/training/augment_training_data.py --phase=both       # do both
    python scripts/training/augment_training_data.py --phase=1 --limit=100  # test run

Output:
    data/processed/augmented_upgraded.jsonl     — phase 1 (improved answers)
    data/processed/augmented_expanded.jsonl     — phase 2 (new Q&A pairs)
    data/processed/master_qa_v3.jsonl           — merged master file
"""

import json
import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from openai import OpenAI
from dotenv import load_dotenv

# Add repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
MASTER_QA  = OUTPUT_DIR / "master_qa.jsonl"

MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 0.3  # seconds between API calls

# Weak categories (scored 2-3/5 in eval)
WEAK_CATEGORIES = {
    "thermal_properties",
    "glass_composition",
    "glass_melting",
    "glass_ceramics",
    "mechanical_properties",
    "glass_science_fundamentals",
}

# Keywords that indicate weak-category questions
WEAK_KEYWORDS = [
    "transition temperature", "tg", "thermal expansion", "cte", "annealing",
    "softening point", "viscosity", "composition", "sio2", "na2o", "b2o3",
    "melting", "fining", "batch", "furnace", "refining",
    "glass-ceramic", "crystallization", "nucleation",
    "modulus", "hardness", "fracture", "strength",
    "bridging oxygen", "network former", "modifier", "nbo",
]

RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("augmentation.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── OpenAI Client ─────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def call_gpt(client: OpenAI, system: str, user: str, temperature: float = 0.7) -> str | None:
    """Call GPT with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            time.sleep(RATE_LIMIT_DELAY)
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"API call failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


# ── Phase 1: Quality Upgrade ─────────────────────────────────────────────────

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


def phase1_upgrade(client: OpenAI, pairs: list, limit: int = 5000) -> list:
    """Upgrade the weakest answers to expert quality."""
    logger.info("=" * 65)
    logger.info("PHASE 1 — Quality Upgrade")
    logger.info("=" * 65)

    # Sort by answer length (shortest first = weakest)
    scored = [(i, len(p["answer"].split())) for i, p in enumerate(pairs)]
    scored.sort(key=lambda x: x[1])

    # Take the shortest answers
    targets = scored[:limit]
    logger.info(f"  Targeting {len(targets)} shortest answers (avg {sum(s[1] for s in targets)/len(targets):.0f} words)")

    # Incremental save file — never lose progress
    checkpoint_path = OUTPUT_DIR / "augmented_upgraded.jsonl"
    upgraded = []
    errors = 0

    # Resume from checkpoint if exists
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    upgraded.append(json.loads(line))
        logger.info(f"  Resuming from checkpoint: {len(upgraded)} already upgraded")
        # Skip already-processed questions
        done_questions = {p["question"] for p in upgraded}
        targets = [(i, wc) for i, wc in targets if pairs[i]["question"] not in done_questions]
        logger.info(f"  Remaining: {len(targets)} to process")

    # Open file in append mode for incremental saves
    with open(checkpoint_path, "a", encoding="utf-8") as save_file:
        for idx, (pair_idx, word_count) in enumerate(targets):
            pair = pairs[pair_idx]
            q = pair["question"]
            a = pair["answer"]

            user_prompt = f"""Question: {q}

Current answer ({word_count} words): {a}

Rewrite this into a detailed, expert-level answer with specific values and glass science principles."""

            result = call_gpt(client, UPGRADE_SYSTEM, user_prompt, temperature=0.5)
            if result:
                try:
                    data = json.loads(result)
                    new_answer = data.get("upgraded_answer", "")
                    if new_answer and len(new_answer.split()) > word_count:
                        new_pair = {
                            "question": q,
                            "answer": new_answer,
                            "source": "upgraded",
                            "original_word_count": word_count,
                        }
                        upgraded.append(new_pair)
                        # Save immediately — never lose progress
                        save_file.write(json.dumps(new_pair, ensure_ascii=False) + "\n")
                        save_file.flush()
                    else:
                        errors += 1
                except json.JSONDecodeError:
                    errors += 1
            else:
                errors += 1

            if (idx + 1) % 50 == 0:
                logger.info(f"  Progress: {idx+1}/{len(targets)} ({len(upgraded)} upgraded, {errors} errors) [auto-saved]")

    logger.info(f"\n  Phase 1 complete: {len(upgraded)} answers upgraded, {errors} errors")
    logger.info(f"  All saved to: {checkpoint_path}")
    return upgraded


# ── Phase 2: Targeted Expansion ──────────────────────────────────────────────

EXPAND_SYSTEM = """You are a glass science expert generating training data.
Given a Q&A pair, generate 3 NEW related Q&A pairs:
1. A VARIATION: same topic but different angle (e.g., different glass type, different application)
2. A FOLLOW-UP: deeper question about something in the answer
3. A COMPARISON: question comparing this concept with a related concept

Rules for answers:
- 150-300 words each
- Include specific numerical values and units
- Reference glass science principles
- Be technically accurate — use textbook values

Return JSON:
{
  "pairs": [
    {"question": "...", "answer": "...", "type": "variation"},
    {"question": "...", "answer": "...", "type": "followup"},
    {"question": "...", "answer": "...", "type": "comparison"}
  ]
}"""


def is_weak_category(question: str) -> bool:
    """Check if a question belongs to a weak category."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in WEAK_KEYWORDS)


def phase2_expand(client: OpenAI, pairs: list, limit: int = 3000) -> list:
    """Generate new Q&A pairs focused on weak categories."""
    logger.info("=" * 65)
    logger.info("PHASE 2 — Targeted Expansion")
    logger.info("=" * 65)

    # Filter to weak-category questions
    weak_pairs = [p for p in pairs if is_weak_category(p["question"])]
    logger.info(f"  Found {len(weak_pairs)} weak-category questions")

    # Sample from weak categories
    random.shuffle(weak_pairs)
    targets = weak_pairs[:limit]
    logger.info(f"  Targeting {len(targets)} pairs for expansion (3x each)")

    # Incremental save file — never lose progress
    checkpoint_path = OUTPUT_DIR / "augmented_expanded.jsonl"
    expanded = []
    errors = 0

    # Resume from checkpoint if exists
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    expanded.append(json.loads(line))
        logger.info(f"  Resuming from checkpoint: {len(expanded)} already expanded")
        # Skip already-processed source questions
        done_questions = {p.get("source_question", "") for p in expanded}
        targets = [p for p in targets if p["question"] not in done_questions]
        logger.info(f"  Remaining: {len(targets)} to process")

    with open(checkpoint_path, "a", encoding="utf-8") as save_file:
        for idx, pair in enumerate(targets):
            q = pair["question"]
            a = pair["answer"][:500]  # truncate long answers for the prompt

            user_prompt = f"""Original Q&A pair:
Q: {q}
A: {a}

Generate 3 new related Q&A pairs (variation, follow-up, comparison)."""

            result = call_gpt(client, EXPAND_SYSTEM, user_prompt, temperature=0.8)
            if result:
                try:
                    data = json.loads(result)
                    new_pairs = data.get("pairs", [])
                    for np_item in new_pairs:
                        nq = np_item.get("question", "")
                        na = np_item.get("answer", "")
                        if nq and na and len(na.split()) >= 10:
                            new_pair = {
                                "question": nq,
                                "answer": na,
                                "source": f"expanded_{np_item.get('type', 'unknown')}",
                                "source_question": q,
                            }
                            expanded.append(new_pair)
                            # Save immediately — never lose progress
                            save_file.write(json.dumps(new_pair, ensure_ascii=False) + "\n")
                            save_file.flush()
                except json.JSONDecodeError:
                    errors += 1
            else:
                errors += 1

            if (idx + 1) % 50 == 0:
                logger.info(f"  Progress: {idx+1}/{len(targets)} ({len(expanded)} new pairs, {errors} errors) [auto-saved]")

    logger.info(f"\n  Phase 2 complete: {len(expanded)} new pairs generated, {errors} errors")
    logger.info(f"  All saved to: {checkpoint_path}")
    return expanded


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smart data augmentation for Glass Expert AI")
    parser.add_argument("--phase", choices=["1", "2", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit pairs per phase (0 = defaults: 5000 for phase 1, 3000 for phase 2)")
    parser.add_argument("--dry-run", action="store_true", help="Just count, don't call API")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("Glass Expert AI — Smart Data Augmentation")
    logger.info(f"Phase: {args.phase}  |  Model: {MODEL}")
    logger.info("=" * 65)

    # Load existing data
    if not MASTER_QA.exists():
        logger.error(f"Master Q&A not found: {MASTER_QA}")
        sys.exit(1)

    pairs = []
    with open(MASTER_QA, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info(f"Loaded {len(pairs):,} existing Q&A pairs")

    if args.dry_run:
        weak = [p for p in pairs if is_weak_category(p["question"])]
        short = sorted(pairs, key=lambda p: len(p["answer"].split()))
        logger.info(f"\nDry run stats:")
        logger.info(f"  Weak-category pairs: {len(weak)}")
        logger.info(f"  Shortest answer: {len(short[0]['answer'].split())} words")
        logger.info(f"  Median answer: {len(short[len(short)//2]['answer'].split())} words")
        logger.info(f"  Phase 1 would upgrade: {min(args.limit or 5000, len(pairs))} answers")
        logger.info(f"  Phase 2 would expand: {min(args.limit or 3000, len(weak))} × 3 = {min(args.limit or 3000, len(weak)) * 3} new pairs")
        return

    client = get_client()

    upgraded_pairs = []
    expanded_pairs = []

    # Phase 1: Quality upgrade
    if args.phase in ("1", "both"):
        p1_limit = args.limit if args.limit > 0 else 5000
        upgraded_pairs = phase1_upgrade(client, pairs, limit=p1_limit)

        out_path = OUTPUT_DIR / "augmented_upgraded.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for p in upgraded_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        logger.info(f"  Saved: {out_path} ({len(upgraded_pairs)} pairs)")

    # Phase 2: Targeted expansion
    if args.phase in ("2", "both"):
        p2_limit = args.limit if args.limit > 0 else 3000
        expanded_pairs = phase2_expand(client, pairs, limit=p2_limit)

        out_path = OUTPUT_DIR / "augmented_expanded.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for p in expanded_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        logger.info(f"  Saved: {out_path} ({len(expanded_pairs)} pairs)")

    # Merge into master v3
    logger.info("\nMerging into master_qa_v3.jsonl...")
    all_pairs = pairs + upgraded_pairs + expanded_pairs
    master_v3_path = OUTPUT_DIR / "master_qa_v3.jsonl"
    with open(master_v3_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"\n{'=' * 65}")
    logger.info("AUGMENTATION COMPLETE")
    logger.info(f"{'=' * 65}")
    logger.info(f"  Original pairs:     {len(pairs):,}")
    logger.info(f"  Upgraded answers:   {len(upgraded_pairs):,}")
    logger.info(f"  New expanded pairs: {len(expanded_pairs):,}")
    logger.info(f"  Total master v3:    {len(all_pairs):,}")
    logger.info(f"  Master file:        {master_v3_path}")
    logger.info(f"{'=' * 65}")
    logger.info("\nNext steps:")
    logger.info("  1. Run prepare_training_data_v3.py to format for training")
    logger.info("  2. Update INPUT_FILES in prepare_training_data_v3.py to include master_qa_v3.jsonl")
    logger.info("  3. Fine-tune with the enlarged dataset")


if __name__ == "__main__":
    main()
