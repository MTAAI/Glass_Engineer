"""
Glass Expert AI — Embedding Migration Script
==============================================
Re-embeds ALL documents in pgvector with the current embedding model.

Use this when switching embedding models (e.g., bge-large-en-v1.5 → bge-m3).
Old and new vectors are incompatible — queries won't match if models differ.

Usage:
    python scripts/migrate_embeddings.py                    # dry run (count only)
    python scripts/migrate_embeddings.py --execute          # actually re-embed
    python scripts/migrate_embeddings.py --execute --batch=64  # larger batches

Steps:
    1. Loads all documents from pgvector (id, content)
    2. Re-embeds content with the currently configured EMBEDDING_MODEL
    3. Updates the embedding column in-place
    4. Rebuilds the IVFFlat index for optimal search performance
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add repo root for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from loguru import logger
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")


def main():
    parser = argparse.ArgumentParser(description="Re-embed all documents with current model")
    parser.add_argument("--execute", action="store_true", help="Actually perform the migration (default: dry run)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for embedding (default: 32)")
    args = parser.parse_args()

    import psycopg2
    from pgvector.psycopg2 import register_vector
    from ingestion.embedder import embed_texts, MODEL_NAME

    logger.info("=" * 65)
    logger.info("Glass Expert AI — Embedding Migration")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Mode:  {'EXECUTE' if args.execute else 'DRY RUN'}")
    logger.info(f"Batch: {args.batch}")
    logger.info("=" * 65)

    # Connect to DB
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL not set")
        sys.exit(1)

    conn = psycopg2.connect(db_url)
    register_vector(conn)
    cur = conn.cursor()

    # Count documents
    cur.execute("SELECT count(*) FROM documents")
    total = cur.fetchone()[0]
    logger.info(f"\nTotal documents: {total:,}")

    if not args.execute:
        logger.info("\nDry run — no changes made. Use --execute to migrate.")
        cur.close()
        conn.close()
        return

    # Load all document IDs and content
    logger.info("\nLoading document content...")
    cur.execute("SELECT id::text, content FROM documents ORDER BY created_at")
    rows = cur.fetchall()
    logger.info(f"Loaded {len(rows):,} documents")

    # Re-embed in batches
    start_time = time.time()
    updated = 0
    errors = 0

    for batch_start in range(0, len(rows), args.batch):
        batch = rows[batch_start:batch_start + args.batch]
        ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]

        try:
            result = embed_texts(texts)
            dense_vecs = result["dense"]

            # Update each document's embedding
            for doc_id, vec in zip(ids, dense_vecs):
                cur.execute(
                    "UPDATE documents SET embedding = %s WHERE id = %s::uuid",
                    (vec.tolist(), doc_id),
                )
            conn.commit()
            updated += len(batch)

            if (batch_start + args.batch) % (args.batch * 10) == 0 or batch_start == 0:
                elapsed = time.time() - start_time
                rate = updated / elapsed if elapsed > 0 else 0
                logger.info(f"  Progress: {updated:,}/{total:,} ({rate:.0f} docs/sec)")

        except Exception as e:
            logger.error(f"  Batch error at offset {batch_start}: {e}")
            conn.rollback()
            errors += len(batch)

    elapsed = time.time() - start_time
    logger.info(f"\nEmbedding update complete: {updated:,} updated, {errors:,} errors ({elapsed:.1f}s)")

    # Rebuild IVFFlat index
    logger.info("\nRebuilding IVFFlat index (this may take a minute)...")
    try:
        cur.execute("REINDEX INDEX documents_embedding_idx")
        conn.commit()
        logger.info("Index rebuilt successfully")
    except Exception as e:
        logger.warning(f"Index rebuild failed: {e}")
        conn.rollback()

    cur.close()
    conn.close()

    logger.info("\n" + "=" * 65)
    logger.info("MIGRATION COMPLETE")
    logger.info(f"  Documents re-embedded: {updated:,}")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Time: {elapsed:.1f}s")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
