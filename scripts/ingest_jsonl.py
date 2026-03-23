"""
Glass Expert AI — JSONL Q&A Ingestion Script
Reads Q&A pairs from master_qa.jsonl and ingests them into pgvector.

Usage:
    python scripts/ingest_jsonl.py
    python scripts/ingest_jsonl.py --batch-size 64 --device cuda
    python scripts/ingest_jsonl.py --limit 100  # test with first 100 pairs
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import numpy as np
from dotenv import load_dotenv
from loguru import logger

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai"
)
MASTER_QA_PATH = (
    Path(__file__).parent.parent / "data" / "processed" / "master_qa.jsonl"
)

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)


def get_db_connection():
    from pgvector.psycopg2 import register_vector
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def load_embedding_model(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    logger.info(f"Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    logger.info("Embedding model loaded.")
    return model


def embed_batch(model, texts: list) -> np.ndarray:
    prefixed = [
        "Represent this sentence for searching relevant passages: " + t
        for t in texts
    ]
    return model.encode(
        prefixed,
        batch_size=len(prefixed),
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def ingest(batch_size: int = 32, device: str = "cpu", limit: int = 0, skip: int = 0):
    if not MASTER_QA_PATH.exists():
        logger.error(f"File not found: {MASTER_QA_PATH}")
        sys.exit(1)

    # Count total lines
    with open(MASTER_QA_PATH, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    if limit > 0:
        total = min(total, limit)
    logger.info(f"Ingesting {total:,} Q&A pairs from {MASTER_QA_PATH.name}")

    model = load_embedding_model(device)
    conn = get_db_connection()
    cur = conn.cursor()

    # Check how many qa_pair chunks already exist
    cur.execute("SELECT COUNT(*) FROM documents WHERE source_type = 'qa_pair';")
    existing = cur.fetchone()[0]
    if existing > 0:
        logger.info(f"Found {existing:,} existing qa_pair chunks — skipping those")

    batch_texts = []
    batch_records = []
    ingested = 0
    skipped = 0
    start_time = time.time()

    with open(MASTER_QA_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if skip > 0 and line_num <= skip:
                continue
            if limit > 0 and line_num > limit:
                break

            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue

            question = record.get("question", "").strip()
            answer = record.get("answer", "").strip()
            if not question or not answer:
                skipped += 1
                continue

            # Format as a single text chunk for embedding
            # Strip NUL bytes — PostgreSQL text columns reject \x00
            question = question.replace("\x00", "")
            answer = answer.replace("\x00", "")
            text = f"Question: {question}\n\nAnswer: {answer}"
            title = question[:120]  # Use question as title (truncated)

            batch_texts.append(text)
            batch_records.append({
                "title": title,
                "source_type": "qa_pair",
                "language": "en",
                "content": text,
                "metadata": json.dumps({
                    "source_file": "master_qa.jsonl",
                    "line_number": line_num,
                    "ingested_at": datetime.utcnow().isoformat(),
                }),
            })

            # Process batch
            if len(batch_texts) >= batch_size:
                embeddings = embed_batch(model, batch_texts)
                for rec, emb in zip(batch_records, embeddings):
                    cur.execute(
                        """
                        INSERT INTO documents (title, source_type, language, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            rec["title"],
                            rec["source_type"],
                            rec["language"],
                            rec["content"],
                            rec["metadata"],
                            emb.tolist(),
                        ),
                    )
                conn.commit()
                ingested += len(batch_texts)

                elapsed = time.time() - start_time
                rate = ingested / elapsed if elapsed > 0 else 0
                eta = (total - ingested) / rate if rate > 0 else 0
                logger.info(
                    f"  {ingested:,}/{total:,} ingested "
                    f"({ingested/total*100:.1f}%) | "
                    f"{rate:.1f} pairs/sec | "
                    f"ETA: {int(eta//60)}m {int(eta%60)}s"
                )

                batch_texts = []
                batch_records = []

    # Final batch
    if batch_texts:
        embeddings = embed_batch(model, batch_texts)
        for rec, emb in zip(batch_records, embeddings):
            cur.execute(
                """
                INSERT INTO documents (title, source_type, language, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    rec["title"],
                    rec["source_type"],
                    rec["language"],
                    rec["content"],
                    rec["metadata"],
                    emb.tolist(),
                ),
            )
        conn.commit()
        ingested += len(batch_texts)

    elapsed = time.time() - start_time
    cur.close()
    conn.close()

    logger.info(f"Done! Ingested {ingested:,} Q&A pairs in {elapsed:.1f}s")
    logger.info(f"Skipped {skipped} invalid records")
    logger.info(f"Rate: {ingested/elapsed:.1f} pairs/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest JSONL Q&A pairs into pgvector")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--limit", type=int, default=0, help="Limit to first N pairs (0=all)")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N lines (for resume)")
    args = parser.parse_args()

    ingest(batch_size=args.batch_size, device=args.device, limit=args.limit, skip=args.skip)
