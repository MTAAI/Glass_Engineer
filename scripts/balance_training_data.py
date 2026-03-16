"""
Balance training data to prevent SciGlass from dominating.
Creates a balanced dataset with controlled proportions.

Target: ~150K total
  - 82K original Q&A (100% - these are highest quality)
  - 11.7K document chunks (100%)
  - ~56K SciGlass (random sample from 413K)

Usage:
    python scripts/balance_training_data.py
"""
import json
import random
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

DATA_DIR = Path("models/qwen14b-glass-expert/data")
SCIGLASS_SAMPLE = 56_000  # cap SciGlass at this many


def load_jsonl(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def main():
    # Load each source
    combined = load_jsonl(DATA_DIR / "combined_train.jsonl")
    chunks = load_jsonl(DATA_DIR / "chunks_train.jsonl")
    sciglass = load_jsonl(DATA_DIR / "sciglass_train.jsonl")

    logger.info(f"Original Q&A:     {len(combined):,}")
    logger.info(f"Document chunks:  {len(chunks):,}")
    logger.info(f"SciGlass:         {len(sciglass):,}")

    # Downsample SciGlass
    if len(sciglass) > SCIGLASS_SAMPLE:
        random.seed(42)
        sciglass = random.sample(sciglass, SCIGLASS_SAMPLE)
        logger.info(f"SciGlass sampled: {len(sciglass):,}")

    # Merge all
    all_data = combined + chunks + sciglass
    random.seed(42)
    random.shuffle(all_data)

    # Write balanced training file
    output_path = DATA_DIR / "balanced_train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_data:
            f.write(line + "\n")

    logger.info(f"\nBalanced training file: {output_path}")
    logger.info(f"Total examples: {len(all_data):,}")

    # Also create a validation set (2% of data, stratified)
    val_size = max(int(len(all_data) * 0.02), 1000)
    random.seed(123)
    val_indices = set(random.sample(range(len(all_data)), val_size))

    train_path = DATA_DIR / "final_train.jsonl"
    val_path = DATA_DIR / "final_val.jsonl"

    train_count = 0
    val_count = 0
    with open(train_path, "w", encoding="utf-8") as tf, \
         open(val_path, "w", encoding="utf-8") as vf:
        for i, line in enumerate(all_data):
            if i in val_indices:
                vf.write(line + "\n")
                val_count += 1
            else:
                tf.write(line + "\n")
                train_count += 1

    logger.info(f"\nFinal split:")
    logger.info(f"  Train: {train_count:,} -> {train_path}")
    logger.info(f"  Val:   {val_count:,} -> {val_path}")


if __name__ == "__main__":
    main()
