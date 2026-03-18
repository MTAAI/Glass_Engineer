"""
Glass Expert AI — JSONL Q&A Pair Ingestion Script
Ingests Q&A pairs from .jsonl files into the RAG PostgreSQL database.

Usage:
    python ingest_jsonl.py --file data/qa_pairs/papers_qa.jsonl
    python ingest_jsonl.py --file data/qa_pairs/textbooks_qa.jsonl
"""
import os
import sys
import json
import argparse
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import psycopg2
from sentence_transformers import SentenceTransformer

from langdetect import detect, LangDetectException


def _detect_language(text: str) -> str:
    """Detect language — returns 'fa' for Farsi/Arabic, 'en' for everything else."""
    try:
        lang = detect(text)
        return "fa" if lang in ("fa", "ar") else "en"
    except LangDetectException:
        return "en"

DB_URL          = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
BATCH_SIZE      = 500

# Instruction prefix — must match how documents were ingested
_PREFIX = "Represent this sentence for searching relevant passages: "


def ingest_qa_file(file_path: str):
    file_path = Path(file_path)
    logger.info(f"Starting Q&A ingestion: {file_path.name}")

    try:
        conn = psycopg2.connect(DB_URL)
        cur  = conn.cursor()
        logger.info("Database connection successful.")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded.")

    total_lines = sum(1 for _ in open(file_path, encoding="utf-8"))
    logger.info(f"Total Q&A pairs to ingest: {total_lines:,}")

    ingested = 0
    skipped  = 0

    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                skipped += 1
                continue

            try:
                qa = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Line {i+1}: invalid JSON — skipped.")
                skipped += 1
                continue

            question = qa.get("question", "").strip()
            answer   = qa.get("answer",   "").strip()
            if not question or not answer:
                skipped += 1
                continue

            # Apply instruction prefix — matches query-time embedding
            text_for_embedding = _PREFIX + question

            try:
                embedding = model.encode(
                    text_for_embedding,
                    normalize_embeddings=True,
                )
                language = _detect_language(question)
                cur.execute(
                    """
                    INSERT INTO documents_bgem3
                        (title, source_type, language, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        question[:500],
                        "qa_pair",
                        "language",
                        answer,
                        json.dumps({
                            "source_file":    file_path.name,
                            "question":       question,
                            "embedding_model": EMBEDDING_MODEL,
                            "language":        language,
                        }),
                        embedding.tolist(),
                    ),
                )
                ingested += 1
                if ingested % BATCH_SIZE == 0:
                    conn.commit()
                    logger.info(f"Progress: {ingested:,} / {total_lines:,} ingested...")

            except Exception as e:
                logger.error(f"Line {i+1}: insert failed — {e}")
                conn.rollback()
                skipped += 1

    conn.commit()
    conn.close()
    logger.success(f"Done: {ingested:,} Q&A pairs ingested, {skipped:,} skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()

    if not Path(args.file).exists():
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    ingest_qa_file(args.file)