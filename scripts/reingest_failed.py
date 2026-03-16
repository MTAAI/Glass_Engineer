"""
Re-ingest documents that previously failed during ingestion.
Reads file paths from ingestion_log WHERE status='failed',
clears the old failed entries, and re-runs ingestion.

Usage:
    python scripts/reingest_failed.py
    python scripts/reingest_failed.py --dry-run   # just list what would be re-ingested
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from loguru import logger
from dotenv import load_dotenv

from ingestion.ingest import ingest_document

load_dotenv()

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")


def get_failed_files():
    """Get list of failed ingestion entries."""
    conn = psycopg2.connect(os.getenv(
        "DATABASE_URL",
        "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai",
    ))
    cur = conn.cursor()
    cur.execute("""
        SELECT id, file_name, file_path, source_type, error_message
        FROM ingestion_log
        WHERE status = 'failed'
        ORDER BY file_name
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def clear_failed_entry(entry_id):
    """Remove a failed ingestion log entry so it can be re-ingested cleanly."""
    conn = psycopg2.connect(os.getenv(
        "DATABASE_URL",
        "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai",
    ))
    cur = conn.cursor()
    cur.execute("DELETE FROM ingestion_log WHERE id = %s", (str(entry_id),))
    conn.commit()
    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Re-ingest failed documents")
    parser.add_argument("--dry-run", action="store_true", help="Just list files, don't ingest")
    args = parser.parse_args()

    failed = get_failed_files()
    logger.info(f"Found {len(failed)} failed ingestion entries")

    if not failed:
        logger.info("Nothing to re-ingest!")
        return

    # Group by error type
    by_error = {}
    for entry_id, file_name, file_path, source_type, error_msg in failed:
        by_error.setdefault(error_msg, []).append(file_name)
    for err, files in by_error.items():
        logger.info(f"  {len(files)}x: {err}")

    if args.dry_run:
        logger.info("Dry run — not ingesting. Remove --dry-run to proceed.")
        return

    success = 0
    still_failed = 0

    for entry_id, file_name, file_path, source_type, error_msg in failed:
        # Try to find the file
        if file_path and Path(file_path).exists():
            path = file_path
        else:
            # Search common data directories
            path = None
            for search_dir in [
                "data/papers/cutting_edge",
                "data/textbooks",
                "data/Textbooks 2",
                "data/source_documents",
                "data/sample_docs",
            ]:
                candidate = Path(search_dir) / file_name
                if candidate.exists():
                    path = str(candidate)
                    break

        if not path:
            logger.warning(f"File not found: {file_name} — skipping")
            still_failed += 1
            continue

        # Clear the old failed entry
        clear_failed_entry(entry_id)

        # Re-ingest
        try:
            source = source_type or "paper"
            chunks = ingest_document(path, source)
            logger.info(f"  OK: {file_name} → {chunks} chunks")
            success += 1
        except Exception as e:
            logger.error(f"  STILL FAILED: {file_name} — {e}")
            still_failed += 1

    logger.info(f"\nRe-ingestion complete: {success} succeeded, {still_failed} still failed")


if __name__ == "__main__":
    main()
