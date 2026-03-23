"""
Glass Expert AI — RAG-Aware Training Data Preparation (v3)
============================================================
Creates training data that matches the INFERENCE format exactly.

Problem: v1/v2 trained on pure Q&A — model never learned to extract
from provided context. At inference it receives RAG context but doesn't
know how to use it → hallucinates from parametric memory instead.

Solution: Generate training examples where user message includes
retrieved context, matching the exact format used at inference time.

Two modes:
  --mode=db       Retrieve real chunks from pgvector (recommended, most realistic)
  --mode=self     Use the answer itself as synthetic context (no DB needed)

Usage:
    python prepare_training_data_v3.py --mode=db      # requires postgres running
    python prepare_training_data_v3.py --mode=self     # offline, no DB needed

Output files (in data/processed/):
    llama_train_v3.jsonl   — RAG-context format, training split
    llama_val_v3.jsonl     — RAG-context format, validation split
"""

import json
import random
import argparse
import logging
import sys
import os
import textwrap
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────

# All Q&A source files (relative to repo root)
INPUT_FILES = [
    "data/processed/master_qa.jsonl",                            # 42,614 merged pairs
    "data/qa_pairs/generated/glass_expert_v3_train.jsonl",       # 10,119 v3 pairs (chat format)
    "data/qa_pairs/generated/glass_expert_detailed_patch.jsonl",  # 530 detailed pairs
]

OUTPUT_DIR = Path("data/processed")
TRAIN_RATIO = 0.90
RANDOM_SEED = 42
RAG_CONTEXT_RATIO = 0.70  # 70% with RAG context, 30% pure Q&A (for robustness)

# Quality thresholds
MIN_QUESTION_LENGTH = 15
MIN_ANSWER_LENGTH = 30
MAX_ANSWER_LENGTH = 5000
MIN_ANSWER_WORDS = 5

# Context settings
MAX_CONTEXT_CHARS = 6000   # Match inference truncation limit in llm.py
TOP_K_RETRIEVE = 5         # Number of chunks to retrieve per question

# System prompt — MUST match LOCAL_SYSTEM_PROMPT in retrieval/llm.py
SYSTEM_PROMPT = (
    "You are Glass Expert AI, a highly specialized assistant trained on thousands of "
    "glass science research papers, textbooks, and material databases. "
    "You provide accurate, detailed, and technically precise answers about glass "
    "composition, properties, manufacturing processes, defects, characterization, "
    "and applications. Always cite relevant glass science principles in your answers."
)

# User message template — MUST match the format in retrieval/llm.py generate_answer()
USER_MESSAGE_WITH_CONTEXT = """Reference information:
{context}

Based on the reference information above, answer this question: {question}

Important: Use specific numbers, temperatures, and compositions from the reference information. Do not make up values."""

# Plain Q&A template (for the 30% without context)
USER_MESSAGE_PLAIN = "{question}"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("data_preparation_v3.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Format normaliser ────────────────────────────────────────────────────────

def normalise_pair(raw: dict) -> dict | None:
    """Extract question/answer from any supported format."""
    # Format A: {"question": "...", "answer": "..."}
    if "question" in raw or "q" in raw:
        q = str(raw.get("question") or raw.get("q", "")).strip()
        a = str(raw.get("answer") or raw.get("a", "")).strip()
        if q and a:
            return {"question": q, "answer": a}
        return None

    # Format B: {"messages": [...]}
    if "messages" in raw and isinstance(raw["messages"], list):
        user_content = ""
        assistant_content = ""
        for msg in raw["messages"]:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()
            if role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content
        if user_content and assistant_content:
            return {"question": user_content, "answer": assistant_content}
    return None


def is_quality(pair: dict) -> bool:
    """Basic quality check."""
    q = pair.get("question", "").strip()
    a = pair.get("answer", "").strip()
    if len(q) < MIN_QUESTION_LENGTH or len(a) < MIN_ANSWER_LENGTH:
        return False
    if len(a) > MAX_ANSWER_LENGTH or len(a.split()) < MIN_ANSWER_WORDS:
        return False
    return True


def normalize_question(q: str) -> str:
    """Lowercase, strip punctuation — used for dedup."""
    import re
    q = q.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def question_hash(q: str) -> str:
    import hashlib
    return hashlib.md5(normalize_question(q).encode()).hexdigest()


# ── Context Generation ────────────────────────────────────────────────────────

def build_self_context(answer: str, all_pairs: list, idx: int) -> str:
    """
    Build synthetic context from the answer itself + random other answers.

    Strategy: The answer IS the context the model should extract from.
    We also add 2-3 distractor passages from other Q&A pairs to teach
    the model to focus on relevant content.
    """
    # Primary context: the answer itself (what the model should extract from)
    context_parts = [answer]

    # Add 2-3 distractor passages from random other pairs
    num_distractors = random.randint(2, 3)
    available = [i for i in range(len(all_pairs)) if i != idx]
    if available:
        distractor_idxs = random.sample(available, min(num_distractors, len(available)))
        for di in distractor_idxs:
            distractor_text = all_pairs[di]["answer"]
            # Truncate long distractors
            if len(distractor_text) > 800:
                distractor_text = distractor_text[:800]
            context_parts.append(distractor_text)

    # Shuffle so the correct answer isn't always first
    random.shuffle(context_parts)

    context = "\n\n".join(context_parts)

    # Truncate to match inference limit
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    return context


def build_db_context(question: str, retriever_func) -> str | None:
    """
    Retrieve real chunks from pgvector and format as plain text.
    Matches the _strip_rag_formatting() output used at inference.
    """
    try:
        results = retriever_func(question, top_k=TOP_K_RETRIEVE, use_cache=False)
        if not results:
            return None

        # Format as plain text (matching _strip_rag_formatting output)
        context_parts = []
        for r in results:
            content = r.get("content", "").strip()
            if content:
                context_parts.append(content)

        context = "\n\n".join(context_parts)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]

        return context if context.strip() else None
    except Exception as e:
        logger.warning(f"DB retrieval failed for question: {e}")
        return None


# ── Training format ───────────────────────────────────────────────────────────

def to_rag_format(question: str, answer: str, context: str) -> dict:
    """Format as Llama 3 chat template with RAG context in user message."""
    user_msg = USER_MESSAGE_WITH_CONTEXT.format(context=context, question=question)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ]
    }


def to_plain_format(question: str, answer: str) -> dict:
    """Format as Llama 3 chat template without context (pure Q&A)."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_MESSAGE_PLAIN.format(question=question)},
            {"role": "assistant", "content": answer},
        ]
    }


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare RAG-aware training data v3")
    parser.add_argument("--mode", choices=["db", "self"], default="self",
                        help="Context source: 'db' = live pgvector, 'self' = synthetic")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of pairs to process (0 = all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    logger.info("=" * 65)
    logger.info("Glass Expert AI — RAG-Aware Training Data Preparation v3")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"RAG context ratio: {RAG_CONTEXT_RATIO:.0%}")
    logger.info("=" * 65)

    # ── Step 1: Load all Q&A pairs ────────────────────────────────────────
    logger.info("\nSTEP 1 — Loading Q&A pairs from all sources")
    all_pairs = []
    seen_hashes = set()
    stats = defaultdict(int)

    for filepath in INPUT_FILES:
        p = REPO_ROOT / filepath
        if not p.exists():
            logger.warning(f"  File not found, skipping: {filepath}")
            continue

        file_count = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    pair = normalise_pair(raw)
                    if pair and is_quality(pair):
                        h = question_hash(pair["question"])
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            all_pairs.append(pair)
                            file_count += 1
                        else:
                            stats["duplicates"] += 1
                except json.JSONDecodeError:
                    stats["parse_errors"] += 1

        logger.info(f"  {p.name}: {file_count:,} unique quality pairs loaded")

    logger.info(f"\n  Total unique pairs: {len(all_pairs):,}")
    logger.info(f"  Duplicates skipped: {stats['duplicates']:,}")

    if args.limit > 0:
        all_pairs = all_pairs[:args.limit]
        logger.info(f"  Limited to: {len(all_pairs):,} pairs")

    # ── Step 2: Setup retriever if DB mode ────────────────────────────────
    retriever_func = None
    if args.mode == "db":
        logger.info("\nSTEP 2 — Connecting to pgvector for real retrieval")
        try:
            from dotenv import load_dotenv
            load_dotenv(REPO_ROOT / ".env")
            from retrieval.retriever import retrieve
            retriever_func = retrieve
            # Test connection
            test_results = retrieve("glass transition temperature", top_k=1, use_cache=False)
            if test_results:
                logger.info(f"  DB connection OK — test query returned {len(test_results)} chunks")
            else:
                logger.warning("  DB returned 0 results for test query — falling back to self mode")
                args.mode = "self"
        except Exception as e:
            logger.error(f"  DB connection failed: {e}")
            logger.info("  Falling back to self mode")
            args.mode = "self"
    else:
        logger.info("\nSTEP 2 — Using self-context mode (no DB needed)")

    # ── Step 3: Generate training examples ────────────────────────────────
    logger.info(f"\nSTEP 3 — Generating {len(all_pairs):,} training examples")

    random.shuffle(all_pairs)
    formatted = []
    context_stats = {"rag": 0, "plain": 0, "db_fail": 0}

    for i, pair in enumerate(all_pairs):
        if (i + 1) % 5000 == 0:
            logger.info(f"  Progress: {i+1:,}/{len(all_pairs):,}")

        q = pair["question"]
        a = pair["answer"]

        # Decide: RAG context or plain Q&A
        use_rag = random.random() < RAG_CONTEXT_RATIO

        if use_rag:
            if args.mode == "db":
                context = build_db_context(q, retriever_func)
                if context:
                    formatted.append(to_rag_format(q, a, context))
                    context_stats["rag"] += 1
                else:
                    # DB returned nothing — use self-context as fallback
                    context = build_self_context(a, all_pairs, i)
                    formatted.append(to_rag_format(q, a, context))
                    context_stats["db_fail"] += 1
            else:
                context = build_self_context(a, all_pairs, i)
                formatted.append(to_rag_format(q, a, context))
                context_stats["rag"] += 1
        else:
            formatted.append(to_plain_format(q, a))
            context_stats["plain"] += 1

    logger.info(f"\n  RAG-context examples: {context_stats['rag']:,}")
    logger.info(f"  Plain Q&A examples:  {context_stats['plain']:,}")
    if context_stats["db_fail"]:
        logger.info(f"  DB retrieval fails:  {context_stats['db_fail']:,} (used self-context)")

    # ── Step 4: Shuffle and split ─────────────────────────────────────────
    logger.info(f"\nSTEP 4 — Shuffling and splitting")
    random.shuffle(formatted)
    split_idx = int(len(formatted) * TRAIN_RATIO)
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    logger.info(f"  Train: {len(train_data):,}")
    logger.info(f"  Val:   {len(val_data):,}")

    # ── Step 5: Save ──────────────────────────────────────────────────────
    logger.info(f"\nSTEP 5 — Saving to {OUTPUT_DIR}")

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        out_path = OUTPUT_DIR / f"llama_{split_name}_v3.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for example in split_data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"  {out_path.name}: {len(split_data):,} examples")

    # ── Step 6: Verify ────────────────────────────────────────────────────
    logger.info(f"\nSTEP 6 — Verification")

    if not train_data:
        logger.error("No training data generated! Check input files exist.")
        sys.exit(1)

    # Check a few examples
    sample = random.choice(train_data)
    user_msg = sample["messages"][1]["content"]
    has_ref = "Reference information:" in user_msg
    logger.info(f"  Sample has context: {has_ref}")
    logger.info(f"  User message length: {len(user_msg)} chars")
    logger.info(f"  System prompt matches: {sample['messages'][0]['content'] == SYSTEM_PROMPT}")

    # Token estimate
    avg_chars = sum(
        sum(len(m["content"]) for m in ex["messages"])
        for ex in train_data[:100]
    ) / min(100, len(train_data))
    avg_tokens = avg_chars / 4  # rough estimate
    logger.info(f"  Avg example length: ~{avg_chars:.0f} chars (~{avg_tokens:.0f} tokens)")

    if avg_tokens > 2048:
        logger.warning(f"  Average exceeds MAX_LENGTH=2048! Consider increasing to 4096.")

    # ── Report ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("RAG-AWARE DATA PREPARATION COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Mode:           {args.mode.upper()}")
    logger.info(f"  Total pairs:    {len(all_pairs):,}")
    logger.info(f"  RAG examples:   {context_stats['rag']:,} ({context_stats['rag']/len(formatted)*100:.0f}%)")
    logger.info(f"  Plain examples: {context_stats['plain']:,} ({context_stats['plain']/len(formatted)*100:.0f}%)")
    logger.info(f"  Train file:     data/processed/llama_train_v3.jsonl ({len(train_data):,})")
    logger.info(f"  Val file:       data/processed/llama_val_v3.jsonl ({len(val_data):,})")
    logger.info("=" * 65)
    logger.info("\nNext steps:")
    logger.info("  1. Review a few examples:  head -1 data/processed/llama_train_v3.jsonl | python -m json.tool")
    logger.info("  2. Fine-tune:  python scripts/training/finetune_glass_llm.py")
    logger.info("     (Update TRAIN_FILE/VAL_FILE to point to v3 files)")


if __name__ == "__main__":
    main()
