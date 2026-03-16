"""
Glass Expert AI — SciGlass CSV Ingestion
Converts the sparse glass_data.csv (276K rows × 4834 columns) into
searchable natural-language chunks for pgvector.

Each row becomes a text like:
  "Glass #20001 — Composition: SiO2 72.5%, Na2O 14.2%, CaO 8.5%.
   Properties: Tg=540°C, density=2.49 g/cm³, refractive index=1.518."

Rows are batched into groups of 10 to create denser chunks for embedding.

Usage:
    python scripts/ingest_sciglass.py
    python scripts/ingest_sciglass.py --batch-size 5 --limit 1000
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from loguru import logger
from dotenv import load_dotenv
from ingestion.embedder import embed_texts

load_dotenv()

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

CSV_PATH = Path("data/csv/glass_data.csv")
BATCH_SIZE = 10  # rows per chunk
EMBED_BATCH = 32


def get_db():
    from pgvector.psycopg2 import register_vector
    conn = psycopg2.connect(os.getenv(
        "DATABASE_URL",
        "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai",
    ))
    register_vector(conn)
    return conn


def load_property_names():
    """Load LISTPROP.csv to map property codes to readable names."""
    prop_path = Path("data/csv/LISTPROP.csv")
    if prop_path.exists():
        df = pd.read_csv(str(prop_path), encoding="utf-8", on_bad_lines="skip")
        # Build mapping from field name to readable name + unit
        mapping = {}
        for _, row in df.iterrows():
            fld = str(row.get("FLDNAM", "")).strip().lower()
            name = str(row.get("NAME", "")).strip()
            unit = str(row.get("Unit", "")).strip()
            if fld and name:
                mapping[fld] = f"{name} ({unit})" if unit and unit != "nan" else name
        return mapping
    return {}


def row_to_text(row, prop_names):
    """Convert a sparse glass data row to natural language."""
    glass_id = row.get("glass_id", "unknown")

    # Extract non-null compositions (comp_* columns)
    comps = []
    props = []

    for col, val in row.items():
        if pd.isna(val) or str(val).strip() == "" or col == "glass_id":
            continue

        val_str = str(val).strip()

        if col.startswith("comp_"):
            # Composition component — e.g. comp_sio2 -> SiO2
            component = col[5:].upper()
            # Clean up common patterns
            component = component.replace("_", "")
            try:
                fval = float(val_str)
                if fval > 0:
                    comps.append(f"{component}: {fval:.2f}%")
            except ValueError:
                comps.append(f"{component}: {val_str}")

        elif col.startswith("prop_") or col.startswith("t_") or col.startswith("d_"):
            # Property value
            readable = prop_names.get(col.lower(), col)
            try:
                fval = float(val_str)
                props.append(f"{readable}={fval:.4g}")
            except ValueError:
                props.append(f"{readable}={val_str}")

    if not comps and not props:
        return None

    parts = [f"Glass #{glass_id}"]
    if comps:
        parts.append(f"Composition: {', '.join(comps[:20])}")  # cap at 20 components
    if props:
        parts.append(f"Properties: {', '.join(props[:15])}")

    return " — ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Ingest SciGlass CSV into pgvector")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Rows per chunk")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0=all)")
    parser.add_argument("--offset", type=int, default=0, help="Start from this row")
    args = parser.parse_args()

    logger.info(f"Loading {CSV_PATH}...")

    # Load property name mapping
    prop_names = load_property_names()
    logger.info(f"Loaded {len(prop_names)} property name mappings")

    # Read CSV in chunks to handle the 879MB file
    chunk_iter = pd.read_csv(
        str(CSV_PATH),
        encoding="utf-8",
        on_bad_lines="skip",
        chunksize=5000,
        low_memory=False,
    )

    conn = get_db()
    cur = conn.cursor()

    total_ingested = 0
    total_chunks = 0
    row_offset = 0

    for df_chunk in chunk_iter:
        for start_idx in range(0, len(df_chunk), args.batch_size):
            if args.offset > 0 and row_offset < args.offset:
                row_offset += args.batch_size
                continue

            batch = df_chunk.iloc[start_idx:start_idx + args.batch_size]

            # Convert each row to text
            texts = []
            for _, row in batch.iterrows():
                text = row_to_text(row, prop_names)
                if text:
                    texts.append(text)

            if not texts:
                row_offset += args.batch_size
                continue

            # Combine into one chunk
            chunk_text = "\n".join(texts)

            # Strip NUL bytes
            chunk_text = chunk_text.replace("\x00", "")

            if len(chunk_text) < 50:
                row_offset += args.batch_size
                continue

            total_chunks += 1

            # Embed in batches
            if total_chunks % EMBED_BATCH == 0 or total_chunks == 1:
                # Accumulate chunks then embed in batch
                pass

            # For simplicity, embed and store one at a time
            # (the embedder handles batching internally)
            try:
                embeddings = embed_texts([chunk_text], return_sparse=False)
                dense_emb = embeddings["dense"][0]

                metadata = {
                    "source": "sciglass",
                    "row_start": row_offset,
                    "row_count": len(texts),
                    "ingested_at": datetime.utcnow().isoformat(),
                }

                cur.execute("""
                    INSERT INTO documents (title, source_type, language, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    f"SciGlass Compositions #{row_offset}-{row_offset + len(texts)}",
                    "standard",  # using 'standard' for database entries
                    "en",
                    chunk_text,
                    json.dumps(metadata),
                    dense_emb.tolist(),
                ))
                conn.commit()
                total_ingested += 1

            except Exception as e:
                logger.error(f"Failed to embed/store chunk at row {row_offset}: {e}")
                conn.rollback()

            row_offset += args.batch_size

            if total_ingested % 100 == 0 and total_ingested > 0:
                logger.info(f"  Progress: {total_ingested} chunks stored ({row_offset:,} rows processed)")

            if args.limit > 0 and row_offset >= args.limit:
                break

        if args.limit > 0 and row_offset >= args.limit:
            break

    # Log ingestion
    cur.execute("""
        INSERT INTO ingestion_log (file_name, file_path, source_type, language, chunk_count, status)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        "glass_data.csv",
        str(CSV_PATH.resolve()),
        "standard",
        "en",
        total_ingested,
        "completed",
    ))
    conn.commit()
    cur.close()
    conn.close()

    logger.info(f"\nSciGlass ingestion complete!")
    logger.info(f"  Rows processed: {row_offset:,}")
    logger.info(f"  Chunks stored: {total_ingested:,}")


if __name__ == "__main__":
    main()
