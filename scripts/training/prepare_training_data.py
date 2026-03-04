"""
Glass Expert AI — Training Data Preparation Script
====================================================
Steps:
  1. Merge all Q&A JSONL files into one master dataset
     Handles TWO input formats automatically:
       Format A: {"question": "...", "answer": "..."}   ← papers/textbooks/sciglass
       Format B: {"messages": [{"role":"user",...}, {"role":"assistant",...}]}  ← glass_finetuning_dataset
  2. Deduplicate by question similarity
  3. Quality filter (remove short/low-quality answers)
  4. Format into Llama 3 chat template AND OpenAI fine-tune format
  5. Split into 90% train / 10% validation

Usage:
    python prepare_training_data.py

Output files (in data/processed/):
    master_qa.jsonl          — merged, deduplicated, filtered (all pairs)
    llama_train.jsonl        — Llama 3 format, training split
    llama_val.jsonl          — Llama 3 format, validation split
    openai_train.jsonl       — OpenAI fine-tune format, training split
    openai_val.jsonl         — OpenAI fine-tune format, validation split
    preparation_report.txt   — full statistics report
"""

import json
import random
import re
import hashlib
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_FILES = [
    "data/qa_pairs/generated/papers_qa.jsonl",
    "data/qa_pairs/generated/textbooks_qa.jsonl",
    "data/qa_pairs/generated/sciglass_qa.jsonl",
    "data/qa_pairs/generated/uci_qa.jsonl",                    # skipped if missing
    "data/qa_pairs/generated/glass_finetuning_dataset.jsonl",  # chat format
]

OUTPUT_DIR = Path("data/processed")

# Quality thresholds
MIN_QUESTION_LENGTH = 15    # characters
MIN_ANSWER_LENGTH   = 30    # characters  (lower for short Tg answers like "Tg = 513 K")
MAX_ANSWER_LENGTH   = 4000  # characters
MIN_ANSWER_WORDS    = 5     # words       (lower for short Tg answers)

# Split ratio
TRAIN_RATIO = 0.90

# Reproducibility
RANDOM_SEED = 42

# System prompt injected into every training example
SYSTEM_PROMPT = (
    "You are Glass Expert AI, a highly specialized assistant trained on thousands of "
    "glass science research papers, textbooks, and material databases. "
    "You provide accurate, detailed, and technically precise answers about glass "
    "composition, properties, manufacturing processes, defects, characterization, "
    "and applications. Always cite relevant glass science principles in your answers."
)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("data_preparation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Format normaliser ──────────────────────────────────────────────────────────

def normalise_pair(raw: dict) -> dict | None:
    """
    Accepts any of the following input formats and returns a unified
    {"question": str, "answer": str} dict.  Returns None if unrecognisable.

    Supported formats:
      A) {"question": "...", "answer": "..."}
      B) {"q": "...", "a": "..."}
      C) {"messages": [{"role": "system", ...},
                        {"role": "user",      "content": "..."},
                        {"role": "assistant", "content": "..."}]}
    """
    # Format A / B — plain question + answer keys
    if "question" in raw or "q" in raw:
        q = str(raw.get("question") or raw.get("q", "")).strip()
        a = str(raw.get("answer")   or raw.get("a",  "")).strip()
        if q and a:
            return {"question": q, "answer": a}
        return None

    # Format C — messages list (chat format)
    if "messages" in raw and isinstance(raw["messages"], list):
        user_content      = ""
        assistant_content = ""
        for msg in raw["messages"]:
            role    = msg.get("role", "")
            content = str(msg.get("content", "")).strip()
            if role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content
        if user_content and assistant_content:
            return {"question": user_content, "answer": assistant_content}
        return None

    return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_question(q: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — used for dedup."""
    q = q.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def question_hash(q: str) -> str:
    return hashlib.md5(normalize_question(q).encode()).hexdigest()


def is_quality(pair: dict) -> tuple[bool, str]:
    """
    Returns (True, '') if the pair passes quality checks,
    or (False, reason) if it should be filtered out.
    """
    q = pair.get("question", "").strip()
    a = pair.get("answer",   "").strip()

    if len(q) < MIN_QUESTION_LENGTH:
        return False, f"question too short ({len(q)} chars)"

    if len(a) < MIN_ANSWER_LENGTH:
        return False, f"answer too short ({len(a)} chars)"

    if len(a) > MAX_ANSWER_LENGTH:
        return False, f"answer too long ({len(a)} chars)"

    if len(a.split()) < MIN_ANSWER_WORDS:
        return False, f"answer has too few words ({len(a.split())})"

    # Reject placeholder / error answers
    bad_patterns = [
        r"i (don't|do not|cannot|can't) know",
        r"i'm not sure",
        r"no (answer|information|data) (available|found|provided)",
        r"^n/?a$",
        r"\berror\b",
        r"\bexception\b",
    ]
    a_lower = a.lower()
    for pat in bad_patterns:
        if re.search(pat, a_lower):
            return False, f"answer matches bad pattern"

    return True, ""


# ── Format converters ──────────────────────────────────────────────────────────

def to_llama3_format(pair: dict) -> dict:
    """
    Llama 3 / Mistral / Qwen2 chat template format.
    Compatible with HuggingFace TRL SFTTrainer (apply_chat_template=True).
    """
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ]
    }


def to_openai_format(pair: dict) -> dict:
    """
    OpenAI fine-tuning format.
    https://platform.openai.com/docs/guides/fine-tuning
    """
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ]
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    stats = {
        "loaded":         0,
        "parse_errors":   0,
        "duplicates":     0,
        "filtered":       0,
        "filter_reasons": defaultdict(int),
        "final":          0,
        "train":          0,
        "val":            0,
        "by_file":        {},
    }

    # ── Step 1: Load & Merge ───────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("STEP 1 — Loading and merging Q&A files")
    logger.info("=" * 65)

    all_pairs = []

    for filepath in INPUT_FILES:
        p = Path(filepath)
        if not p.exists():
            logger.warning(f"  ⚠  File not found, skipping: {filepath}")
            continue

        file_loaded = 0
        file_errors = 0

        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw  = json.loads(line)
                    pair = normalise_pair(raw)
                    if pair:
                        all_pairs.append(pair)
                        file_loaded += 1
                    else:
                        file_errors += 1
                except json.JSONDecodeError:
                    file_errors += 1

        stats["by_file"][p.name] = file_loaded
        stats["parse_errors"] += file_errors
        logger.info(f"  ✅  {p.name}: {file_loaded:,} pairs loaded"
                    + (f"  ({file_errors} parse errors)" if file_errors else ""))

    stats["loaded"] = len(all_pairs)
    logger.info(f"\n  Total loaded: {stats['loaded']:,} pairs")

    # ── Step 2: Deduplicate ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("STEP 2 — Deduplicating by question hash")
    logger.info("=" * 65)

    seen_hashes  = set()
    unique_pairs = []

    for pair in all_pairs:
        h = question_hash(pair["question"])
        if h in seen_hashes:
            stats["duplicates"] += 1
        else:
            seen_hashes.add(h)
            unique_pairs.append(pair)

    logger.info(f"  Duplicates removed: {stats['duplicates']:,}")
    logger.info(f"  Unique pairs:       {len(unique_pairs):,}")

    # ── Step 3: Quality Filter ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("STEP 3 — Quality filtering")
    logger.info("=" * 65)

    clean_pairs = []
    for pair in unique_pairs:
        ok, reason = is_quality(pair)
        if ok:
            clean_pairs.append(pair)
        else:
            stats["filtered"] += 1
            stats["filter_reasons"][reason.split("(")[0].strip()] += 1

    logger.info(f"  Pairs removed: {stats['filtered']:,}")
    for reason, cnt in sorted(stats["filter_reasons"].items(), key=lambda x: -x[1]):
        logger.info(f"    • {reason}: {cnt:,}")
    logger.info(f"  Clean pairs:   {len(clean_pairs):,}")

    stats["final"] = len(clean_pairs)

    # ── Step 4: Save master JSONL ──────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("STEP 4 — Saving master Q&A file")
    logger.info("=" * 65)

    master_path = OUTPUT_DIR / "master_qa.jsonl"
    with open(master_path, "w", encoding="utf-8") as f:
        for pair in clean_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info(f"  💾  {master_path}  ({stats['final']:,} pairs)")

    # ── Step 5: Shuffle & Split ────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("STEP 5 — Shuffling and splitting train / validation")
    logger.info("=" * 65)

    random.shuffle(clean_pairs)
    split_idx   = int(len(clean_pairs) * TRAIN_RATIO)
    train_pairs = clean_pairs[:split_idx]
    val_pairs   = clean_pairs[split_idx:]

    stats["train"] = len(train_pairs)
    stats["val"]   = len(val_pairs)
    logger.info(f"  Train: {stats['train']:,} pairs  ({TRAIN_RATIO*100:.0f}%)")
    logger.info(f"  Val:   {stats['val']:,} pairs  ({(1-TRAIN_RATIO)*100:.0f}%)")

    # ── Step 6: Save Llama 3 format ───────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("STEP 6 — Saving Llama 3 / Mistral / Qwen chat format")
    logger.info("=" * 65)

    for split_name, split_data in [("train", train_pairs), ("val", val_pairs)]:
        out_path = OUTPUT_DIR / f"llama_{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for pair in split_data:
                f.write(json.dumps(to_llama3_format(pair), ensure_ascii=False) + "\n")
        logger.info(f"  💾  {out_path}  ({len(split_data):,} examples)")

    # ── Step 7: Save OpenAI format ────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("STEP 7 — Saving OpenAI fine-tune format")
    logger.info("=" * 65)

    for split_name, split_data in [("train", train_pairs), ("val", val_pairs)]:
        out_path = OUTPUT_DIR / f"openai_{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for pair in split_data:
                f.write(json.dumps(to_openai_format(pair), ensure_ascii=False) + "\n")
        logger.info(f"  💾  {out_path}  ({len(split_data):,} examples)")

    # ── Step 8: Report ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("STEP 8 — Final Report")
    logger.info("=" * 65)

    report_lines = [
        "Glass Expert AI — Data Preparation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        "Source files:",
    ]
    for fname, cnt in stats["by_file"].items():
        report_lines.append(f"  {fname:<45} {cnt:>8,} pairs")

    report_lines += [
        "",
        "=" * 65,
        f"Total loaded                   : {stats['loaded']:>8,}",
        f"Parse errors                   : {stats['parse_errors']:>8,}",
        f"Duplicates removed             : {stats['duplicates']:>8,}",
        f"Pairs removed by quality filter: {stats['filtered']:>8,}",
        f"Final clean pairs              : {stats['final']:>8,}",
        "-" * 65,
        f"Training set  (90%)            : {stats['train']:>8,}",
        f"Validation set (10%)           : {stats['val']:>8,}",
        "=" * 65,
        "",
        "Output files:",
        "  data/processed/master_qa.jsonl      — all clean pairs (raw)",
        "  data/processed/llama_train.jsonl    — Llama 3 / Mistral / Qwen train",
        "  data/processed/llama_val.jsonl      — Llama 3 / Mistral / Qwen validation",
        "  data/processed/openai_train.jsonl   — OpenAI gpt-4o-mini train",
        "  data/processed/openai_val.jsonl     — OpenAI gpt-4o-mini validation",
        "",
        "Quality filter breakdown:",
    ]
    for reason, cnt in sorted(stats["filter_reasons"].items(), key=lambda x: -x[1]):
        report_lines.append(f"  • {reason}: {cnt:,}")

    report_text = "\n".join(report_lines)
    logger.info("\n" + report_text)

    report_path = OUTPUT_DIR / "preparation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    logger.info(f"\n  📄  Report saved: {report_path}")
    logger.info("\n✅  Data preparation complete!")
    logger.info("    → Use  llama_train.jsonl  for Llama / Mistral / Qwen fine-tuning")
    logger.info("    → Use  openai_train.jsonl for OpenAI gpt-4o-mini fine-tuning")


if __name__ == "__main__":
    main()
