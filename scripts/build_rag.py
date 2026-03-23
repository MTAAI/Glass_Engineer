"""
Glass Expert AI — 03_build_rag.py
Engineer A's script to build the RAG vector database from all available documents.

Usage:
    # Ingest a single file
    python 03_build_rag.py --file data/sample_docs/01_glass_transition_thermal_properties.pdf --type textbook

    # Ingest an entire folder (recommended for bulk ingestion)
    python 03_build_rag.py --folder data/sample_docs/ --type textbook

    # Ingest your real 15GB dataset overnight
    python 03_build_rag.py --folder data/real_data/textbooks/ --type textbook
    python 03_build_rag.py --folder data/real_data/papers/ --type paper
    python 03_build_rag.py --folder data/real_data/qa_pairs/ --type qa_pair

    # Check database stats
    python 03_build_rag.py --stats

    # Clear all data and start fresh
    python 03_build_rag.py --clear
"""
import os
import sys
import argparse
import psycopg2
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from ingestion.ingest import ingest_document, ingest_folder
from ingestion.extractor import SUPPORTED_EXTENSIONS

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/rag_build.log",
    rotation="50 MB",
    retention="30 days",
    level="DEBUG",
)


def get_db_stats() -> dict:
    """Get current database statistics."""
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM documents;")
        total_chunks = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT title) FROM documents;")
        total_docs = cur.fetchone()[0]

        cur.execute("SELECT source_type, COUNT(*) FROM documents GROUP BY source_type ORDER BY COUNT(*) DESC;")
        by_type = cur.fetchall()

        cur.execute("SELECT language, COUNT(*) FROM documents GROUP BY language;")
        by_lang = cur.fetchall()

        cur.execute("SELECT title, COUNT(*) as chunks FROM documents GROUP BY title ORDER BY chunks DESC LIMIT 10;")
        top_docs = cur.fetchall()

        conn.close()
        return {
            "total_chunks": total_chunks,
            "total_docs": total_docs,
            "by_type": by_type,
            "by_lang": by_lang,
            "top_docs": top_docs,
        }
    except Exception as e:
        logger.error(f"Could not connect to database: {e}")
        logger.error("Make sure Docker is running: cd docker && docker compose up -d")
        return None


def print_stats():
    """Print a formatted database statistics report."""
    stats = get_db_stats()
    if not stats:
        return

    print("\n" + "=" * 60)
    print("  Glass Expert AI — RAG Database Statistics")
    print("=" * 60)
    print(f"  Total chunks stored : {stats['total_chunks']:,}")
    print(f"  Total documents     : {stats['total_docs']:,}")
    print()

    print("  By source type:")
    for stype, count in stats["by_type"]:
        print(f"    {stype:<20} {count:>8,} chunks")

    print()
    print("  By language:")
    for lang, count in stats["by_lang"]:
        label = "English" if lang == "en" else "Farsi/Persian"
        print(f"    {label:<20} {count:>8,} chunks")

    print()
    print("  Top 10 documents by chunk count:")
    for title, chunks in stats["top_docs"]:
        print(f"    {title[:45]:<46} {chunks:>5} chunks")

    print("=" * 60 + "\n")


def clear_database():
    """Remove all ingested data from the database."""
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE documents, ingestion_log RESTART IDENTITY CASCADE;")
        conn.commit()
        conn.close()
        logger.info("✅ Database cleared. All chunks and ingestion logs removed.")
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Glass Expert AI — RAG Vector Database Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file",   type=str, help="Path to a single file to ingest")
    group.add_argument("--folder", type=str, help="Path to a folder to ingest recursively")
    group.add_argument("--stats",  action="store_true", help="Show database statistics")
    group.add_argument("--clear",  action="store_true", help="Clear all data from the database")

    parser.add_argument(
        "--type",
        type=str,
        default="textbook",
        choices=["textbook", "paper", "sop", "qa_pair", "manual", "standard"],
        help="Source type for ingested documents (default: textbook)",
    )

    args = parser.parse_args()

    # ── Show stats ─────────────────────────────────────────────────────────────
    if args.stats:
        print_stats()
        return

    # ── Clear database ─────────────────────────────────────────────────────────
    if args.clear:
        confirm = input("⚠️  This will delete ALL ingested data. Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            clear_database()
        else:
            logger.info("Clear cancelled.")
        return

    # ── Ingest single file ─────────────────────────────────────────────────────
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported file type: {file_path.suffix}")
            logger.error(f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}")
            sys.exit(1)

        logger.info(f"Ingesting single file: {file_path.name}")
        start = datetime.now()
        ingest_document(str(file_path), args.type)
        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"Completed in {elapsed:.1f}s")
        print_stats()
        return

    # ── Ingest folder ──────────────────────────────────────────────────────────
    if args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists():
            logger.error(f"Folder not found: {args.folder}")
            sys.exit(1)

        # Count files first
        all_files = [
            f for f in folder_path.rglob("*")
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not all_files:
            logger.warning(f"No supported files found in: {args.folder}")
            logger.warning(f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}")
            return

        logger.info(f"Found {len(all_files)} files to ingest from: {folder_path.name}/")
        logger.info(f"Source type: {args.type}")
        logger.info("Starting ingestion... (this may take a while for large datasets)")

        start = datetime.now()
        ingest_folder(str(folder_path), args.type)
        elapsed = (datetime.now() - start).total_seconds()

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        time_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"

        logger.info(f"Folder ingestion complete in {time_str}")
        print_stats()
        return

    # ── No arguments — show help and stats ────────────────────────────────────
    parser.print_help()
    print()
    print_stats()


if __name__ == "__main__":
    main()
