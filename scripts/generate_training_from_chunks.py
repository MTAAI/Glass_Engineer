"""
Glass Expert AI — Generate Training Q&A from Document Chunks
=============================================================
Reads document chunks from pgvector (papers, textbooks, SciGlass, manuals)
and generates Q&A training pairs in chat format for fine-tuning.

For SciGlass composition data: generates property-lookup Q&A pairs.
For document chunks: generates extractive Q&A pairs from the content.

Output: JSONL in the same format as combined_train.jsonl

Usage:
    python scripts/generate_training_from_chunks.py
    python scripts/generate_training_from_chunks.py --limit 5000
"""
import os
import sys
import json
import re
import random
import argparse
from pathlib import Path
from datetime import datetime

import psycopg2
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

SYSTEM_PROMPT = (
    "You are Glass Expert AI, a highly specialized assistant trained on thousands "
    "of glass science research papers, textbooks, and material databases. You provide "
    "accurate, detailed, and technically precise answers about glass composition, "
    "properties, manufacturing processes, defects, characterization, and applications. "
    "Always cite relevant glass science principles in your answers."
)

SYSTEM_PROMPT_FA = (
    "شما Glass Expert AI هستید، یک دستیار بسیار تخصصی که بر روی هزاران مقاله علمی، "
    "کتاب درسی و پایگاه‌های داده مواد شیشه‌ای آموزش دیده‌اید. شما پاسخ‌های دقیق، "
    "مفصل و از نظر فنی صحیح درباره ترکیب، خواص، فرآیندهای تولید، عیوب، مشخصه‌سازی "
    "و کاربردهای شیشه ارائه می‌دهید."
)

OUTPUT_DIR = Path("models/qwen14b-glass-expert/data")


def get_db():
    return psycopg2.connect(os.getenv(
        "DATABASE_URL",
        "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai",
    ))


# ── Question templates for document chunks ──────────────────────────────────

QUESTION_TEMPLATES = [
    "What does the research say about {topic}?",
    "Explain the key findings regarding {topic}.",
    "Summarize the information about {topic} from the glass science literature.",
    "What are the important aspects of {topic} in glass technology?",
    "Describe {topic} as discussed in glass science research.",
    "What is known about {topic} in the context of glass materials?",
    "How does {topic} relate to glass properties and applications?",
    "What are the main points about {topic} from glass science studies?",
]

QUESTION_TEMPLATES_FA = [
    "درباره {topic} در علم شیشه چه اطلاعاتی وجود دارد؟",
    "یافته‌های کلیدی درباره {topic} را توضیح دهید.",
    "اطلاعات مربوط به {topic} را خلاصه کنید.",
    "جنبه‌های مهم {topic} در فناوری شیشه چیست؟",
]

SCIGLASS_TEMPLATES = [
    "What is the composition of glass #{glass_id}?",
    "Describe the properties of glass #{glass_id}.",
    "What are the components and properties of glass #{glass_id}?",
    "Tell me about the composition and characteristics of glass #{glass_id}.",
]


def extract_topic(content: str) -> str:
    """Extract a topic phrase from document content."""
    # Take first meaningful sentence as topic
    sentences = re.split(r'[.!?]\s+', content[:500])
    for s in sentences:
        s = s.strip()
        if len(s) > 20 and len(s) < 200:
            # Remove common prefixes
            s = re.sub(r'^(Abstract|Introduction|Background|In this|This paper|We)\s+', '', s, flags=re.IGNORECASE)
            return s[:150]
    # Fallback: use first 100 chars
    return content[:100].strip()


def chunk_to_qa(title: str, content: str, source_type: str, language: str) -> dict | None:
    """Convert a document chunk to a Q&A training pair."""
    content = content.strip()
    if len(content) < 100:
        return None

    if language == "fa":
        topic = extract_topic(content)
        question = random.choice(QUESTION_TEMPLATES_FA).format(topic=topic)
        system = SYSTEM_PROMPT_FA
    else:
        topic = extract_topic(content)
        question = random.choice(QUESTION_TEMPLATES).format(topic=topic)
        system = SYSTEM_PROMPT

    # Build answer from content with source attribution
    source_label = {
        "paper": "research paper",
        "textbook": "textbook",
        "manual": "technical manual",
        "sop": "standard operating procedure",
        "standard": "glass composition database",
    }.get(source_type, "reference material")

    answer = f"Based on the {source_label} \"{title}\":\n\n{content[:1500]}"

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def sciglass_to_qa(content: str, title: str) -> dict | None:
    """Convert a SciGlass composition chunk to Q&A pairs."""
    content = content.strip()
    if len(content) < 50:
        return None

    # Extract glass IDs from content
    glass_ids = re.findall(r'Glass #(\d+)', content)
    if glass_ids:
        glass_id = glass_ids[0]
        question = random.choice(SCIGLASS_TEMPLATES).format(glass_id=glass_id)
    else:
        question = "What glass compositions and properties are shown in this SciGlass database entry?"

    answer = f"From the SciGlass database:\n\n{content[:1500]}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max chunks to process")
    parser.add_argument("--output", type=str, default="", help="Output file path")
    args = parser.parse_args()

    conn = get_db()
    cur = conn.cursor()

    # Get all non-qa_pair document chunks
    query = """
        SELECT title, content, source_type, language
        FROM documents
        WHERE source_type != 'qa_pair'
        ORDER BY source_type, title
    """
    if args.limit > 0:
        query += f" LIMIT {args.limit}"

    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    logger.info(f"Loaded {len(rows):,} document chunks for Q&A generation")

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "chunks_train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for title, content, source_type, language in rows:
            if source_type == "standard":
                qa = sciglass_to_qa(content, title)
            else:
                qa = chunk_to_qa(title, content, source_type, language)

            if qa:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                generated += 1
            else:
                skipped += 1

            if generated % 1000 == 0 and generated > 0:
                logger.info(f"  Generated {generated:,} Q&A pairs...")

    logger.info(f"\nQ&A generation complete!")
    logger.info(f"  Generated: {generated:,}")
    logger.info(f"  Skipped: {skipped:,}")
    logger.info(f"  Output: {output_path}")

    # Now merge with existing training data
    existing_train = OUTPUT_DIR / "combined_train.jsonl"
    merged_path = OUTPUT_DIR / "merged_train.jsonl"

    if existing_train.exists():
        logger.info(f"\nMerging with existing training data: {existing_train}")
        existing_count = 0
        with open(merged_path, "w", encoding="utf-8") as out:
            # Copy existing
            with open(existing_train, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    existing_count += 1
            # Append new
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)

        total = existing_count + generated
        logger.info(f"  Existing: {existing_count:,}")
        logger.info(f"  New: {generated:,}")
        logger.info(f"  Merged total: {total:,}")
        logger.info(f"  Merged file: {merged_path}")
    else:
        logger.info("No existing training file found — using generated data only")


if __name__ == "__main__":
    main()
