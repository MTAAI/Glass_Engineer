"""
Glass Expert AI — Main Ingestion Pipeline
Runs the full pipeline: Extract → Chunk → Embed → Store

Usage:
    python ingest.py path/to/document.pdf --type textbook
    python ingest.py path/to/folder/ --type paper
"""

import os
import sys
import json
import argparse
import psycopg2
from pathlib import Path
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.extractor import extract_file, SUPPORTED_EXTENSIONS
from ingestion.chunker   import chunk_text
from ingestion.embedder  import embed_texts

load_dotenv()

# ── Logging setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/ingestion.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
)


# ── Database helpers ───────────────────────────────────────────────────────────
def get_db_connection():
    """Create a PostgreSQL connection with pgvector support."""
    try:
        from pgvector.psycopg2 import register_vector
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        register_vector(conn)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("Make sure Docker containers are running: cd docker && docker compose up -d")
        raise


def store_chunks(
    conn,
    chunks: list[str],
    embeddings: dict,
    doc_info: dict,
    source_type: str,
) -> int:
    """Insert document chunks and their embeddings into PostgreSQL."""
    cur = conn.cursor()
    inserted = 0

    for i, (chunk, dense_emb) in enumerate(zip(chunks, embeddings["dense"])):
        metadata = {
            "chunk_index":   i,
            "total_chunks":  len(chunks),
            "file_path":     doc_info["file_path"],
            "page_count":    doc_info["page_count"],
            "ingested_at":   datetime.utcnow().isoformat(),
        }

        cur.execute(
            """
            INSERT INTO documents
                (title, source_type, language, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                doc_info["title"],
                source_type,
                doc_info["language"],
                chunk,
                json.dumps(metadata),
                dense_emb.tolist(),
            ),
        )
        inserted += 1

    conn.commit()
    cur.close()
    return inserted


def log_ingestion(conn, doc_info: dict, source_type: str, chunk_count: int, status: str = "completed", error: str = None):
    """Record ingestion in the ingestion_log table."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ingestion_log
            (file_name, file_path, source_type, language, chunk_count, status, error_message)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            Path(doc_info["file_path"]).name,
            doc_info["file_path"],
            source_type,
            doc_info.get("language", "en"),
            chunk_count,
            status,
            error,
        ),
    )
    conn.commit()
    cur.close()


# ── Main pipeline ──────────────────────────────────────────────────────────────
def ingest_document(file_path: str, source_type: str = "textbook") -> int:
    """
    Run the full ingestion pipeline for a single document (PDF, CSV, NPZ, TXT, DOCX, JSON).

    Args:
        file_path:   Path to the PDF file.
        source_type: One of: textbook, paper, sop, qa_pair, manual, standard

    Returns:
        Number of chunks inserted into the database.
    """
    logger.info(f"{'='*60}")
    logger.info(f"Starting ingestion: {Path(file_path).name}")
    logger.info(f"Source type: {source_type}")
    logger.info(f"{'='*60}")

    conn = get_db_connection()
    doc_info = {"file_path": file_path, "title": Path(file_path).stem}

    try:
        # ── Step 1: Extract ────────────────────────────────────────────────────
        logger.info(f"Step 1/4 — Extracting text from {Path(file_path).suffix.upper()} file...")
        doc_info = extract_file(file_path)

        # ── Step 2: Chunk ──────────────────────────────────────────────────────
        logger.info("Step 2/4 — Chunking text...")
        chunk_size = int(os.getenv("CHUNK_SIZE", 512))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 64))
        chunks = chunk_text(
            doc_info["text"],
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            language=doc_info["language"],
        )

        if not chunks:
            logger.warning("No chunks created — document may be empty or image-only.")
            log_ingestion(conn, doc_info, source_type, 0, "failed", "No text extracted")
            return 0

        # ── Step 3: Embed ──────────────────────────────────────────────────────
        logger.info(f"Step 3/4 — Generating embeddings for {len(chunks)} chunks...")
        embeddings = embed_texts(chunks, return_sparse=False)

        # ── Step 4: Store ──────────────────────────────────────────────────────
        logger.info("Step 4/4 — Storing in PostgreSQL...")
        inserted = store_chunks(conn, chunks, embeddings, doc_info, source_type)
        log_ingestion(conn, doc_info, source_type, inserted)

        logger.info(f"✅ Ingestion complete: {inserted} chunks stored for '{doc_info['title']}'")
        return inserted

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        log_ingestion(conn, doc_info, source_type, 0, "failed", str(e))
        raise

    finally:
        conn.close()


def ingest_folder(folder_path: str, source_type: str = "textbook") -> dict:
    """Ingest all PDFs in a folder."""
    folder = Path(folder_path)
    all_files = [f for f in folder.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    pdf_files = all_files

    if not pdf_files:
        logger.warning(f"No supported files found in: {folder_path}")
        return {}

    logger.info(f"Found {len(pdf_files)} files to ingest ({", ".join(SUPPORTED_EXTENSIONS)})")
    results = {}

    for pdf_file in pdf_files:
        try:
            count = ingest_document(str(pdf_file), source_type)
            results[pdf_file.name] = {"status": "success", "chunks": count}
        except Exception as e:
            results[pdf_file.name] = {"status": "failed", "error": str(e)}

    # Summary
    success = sum(1 for r in results.values() if r["status"] == "success")
    total_chunks = sum(r.get("chunks", 0) for r in results.values())
    logger.info(f"\n{'='*60}")
    logger.info(f"Folder ingestion complete: {success}/{len(pdf_files)} files successful")
    logger.info(f"Total chunks stored: {total_chunks:,}")
    return results


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Glass Expert AI — Document Ingestion Pipeline"
    )
    parser.add_argument(
        "path",
        help="Path to a PDF file or folder containing PDFs",
    )
    parser.add_argument(
        "--type",
        default="textbook",
        choices=["textbook", "paper", "sop", "qa_pair", "manual", "standard"],
        help="Type of document being ingested (default: textbook)",
    )
    args = parser.parse_args()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    target = Path(args.path)
    if target.is_dir():
        ingest_folder(str(target), args.type)
    elif target.is_file():
        ingest_document(str(target), args.type)
    else:
        logger.error(f"Path not found: {args.path}")
        sys.exit(1)
