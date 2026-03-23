"""
Glass Expert AI — Batched JSONL Ingestion with Quality Filtering
================================================================
Ingests JSONL files into the knowledge base with:
  1. Quality filtering (min length, glass-relevance check, dedup)
  2. Batch processing (configurable batch size, resume support)
  3. Source categorization (auto-detect source type from content)
  4. Dry-run mode (preview what would be ingested)

Usage:
    # Dry run — see what passes filters
    python scripts/ingest_jsonl_batched.py data/qa_pairs/generated/papers_qa.jsonl --dry-run

    # Ingest with defaults (batch=100, source_type=qa_pair)
    python scripts/ingest_jsonl_batched.py data/qa_pairs/generated/papers_qa.jsonl

    # Ingest specific batch range (resume after crash)
    python scripts/ingest_jsonl_batched.py data/processed/master_qa.jsonl --start 5000 --limit 2000

    # Ingest with custom source type and stricter filtering
    python scripts/ingest_jsonl_batched.py data/processed/augmented_all_43k.jsonl --source-type qa_pair --min-words 30 --batch-size 50

    # Skip glass-relevance check (for trusted files)
    python scripts/ingest_jsonl_batched.py data/qa_pairs/generated/textbooks_qa.jsonl --no-relevance-check
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Glass science relevance keywords ─────────────────────────────────────────
GLASS_KEYWORDS = {
    # Core glass terms
    "glass", "silica", "sio2", "borosilicate", "soda-lime", "float glass",
    "tempered", "annealing", "fused silica", "glass transition", "tg ",
    # Composition
    "na2o", "cao", "b2o3", "al2o3", "mgo", "k2o", "bao", "pbo", "tio2",
    "zno", "fe2o3", "p2o5", "li2o", "zro2", "sno2",
    "network former", "network modifier", "intermediate oxide",
    # Properties
    "viscosity", "thermal expansion", "cte", "refractive index", "abbe",
    "young's modulus", "hardness", "fracture", "brittleness",
    "devitrification", "crystallization", "nucleation",
    # Manufacturing
    "melting", "fining", "batch", "cullet", "furnace", "float process",
    "forming", "annealer", "lehr", "tempering", "laminating",
    # Defects
    "seed", "blister", "stone", "cord", "striae", "inclusion",
    "nickel sulfide", "nis", "breakage",
    # Types
    "borosilicate", "aluminosilicate", "lead glass", "crystal glass",
    "optical glass", "fiberglass", "glass fiber", "glass ceramic",
    "e-glass", "s-glass", "container glass", "flat glass",
    # Science
    "zachariasen", "vogel-fulcher", "vft", "angell", "fragility",
    "non-bridging oxygen", "nbo", "bridging oxygen",
    # Farsi terms
    "شیشه", "سیلیکا", "بوروسیلیکات", "آنیلینگ", "سکوریت", "ذوب",
    "ویسکوزیته", "ضریب انبساط", "شکست",
}


def is_glass_relevant(text: str, threshold: int = 2) -> bool:
    """Check if text contains enough glass-science keywords."""
    text_lower = text.lower()
    hits = sum(1 for kw in GLASS_KEYWORDS if kw in text_lower)
    return hits >= threshold


def content_hash(text: str) -> str:
    """MD5 hash for deduplication."""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(text.split())


def parse_jsonl_entry(line: str, line_num: int) -> dict | None:
    """Parse a single JSONL line, return None if invalid."""
    try:
        entry = json.loads(line.strip())
        if not isinstance(entry, dict):
            return None
        return entry
    except (json.JSONDecodeError, ValueError):
        return None


def extract_content(entry: dict) -> str:
    """Extract the main text content from various JSONL formats."""
    # Format 1: question + answer (QA pairs)
    if "question" in entry and "answer" in entry:
        q = entry["question"].strip()
        a = entry["answer"].strip()
        return f"Question: {q}\n\nAnswer: {a}"

    # Format 2: content field
    if "content" in entry:
        return entry["content"].strip()

    # Format 3: text field
    if "text" in entry:
        return entry["text"].strip()

    # Format 4: instruction/input/output (training format)
    if "instruction" in entry:
        parts = [entry["instruction"]]
        if entry.get("input"):
            parts.append(entry["input"])
        if entry.get("output"):
            parts.append(entry["output"])
        return "\n\n".join(parts).strip()

    return ""


def extract_title(entry: dict, content: str) -> str:
    """Extract or generate a title for the entry."""
    # Try explicit title/id fields
    for key in ("title", "id", "source", "name"):
        if key in entry and entry[key]:
            return str(entry[key])[:200]

    # Try question as title (for QA pairs)
    if "question" in entry:
        return entry["question"][:200]

    # Fallback: first 80 chars of content
    return content[:80].replace("\n", " ")


def main():
    parser = argparse.ArgumentParser(description="Batched JSONL ingestion with quality filtering")
    parser.add_argument("file", help="Path to JSONL file")
    parser.add_argument("--source-type", default="qa_pair",
                        choices=["textbook", "paper", "sop", "qa_pair", "manual", "standard"],
                        help="Source type label (default: qa_pair)")
    parser.add_argument("--language", default="en", choices=["en", "fa"],
                        help="Language label (default: en)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Entries per DB commit batch (default: 100)")
    parser.add_argument("--min-words", type=int, default=20,
                        help="Minimum word count to accept (default: 20)")
    parser.add_argument("--max-words", type=int, default=2000,
                        help="Maximum word count (default: 2000)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start from line N (for resume, default: 0)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max entries to process (0=all)")
    parser.add_argument("--no-relevance-check", action="store_true",
                        help="Skip glass-relevance keyword check")
    parser.add_argument("--dedup", action="store_true", default=True,
                        help="Skip duplicates already in DB (default: True)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable deduplication")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only — don't insert into DB")
    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    # Count total lines
    with open(filepath, encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"File: {filepath.name} ({total_lines:,} lines)")

    # Load existing content hashes for dedup
    existing_hashes = set()
    if args.dedup and not args.no_dedup and not args.dry_run:
        print("Loading existing content hashes for deduplication...")
        import psycopg2
        conn = psycopg2.connect(os.getenv(
            "DATABASE_URL",
            "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai"
        ))
        cur = conn.cursor()
        cur.execute("SELECT md5(lower(trim(content))) FROM documents")
        existing_hashes = {r[0] for r in cur.fetchall()}
        cur.close()
        conn.close()
        print(f"  Found {len(existing_hashes):,} existing hashes")

    # ── Process entries ───────────────────────────────────────────────────────
    stats = {
        "total_read": 0,
        "skipped_parse": 0,
        "skipped_empty": 0,
        "skipped_short": 0,
        "skipped_long": 0,
        "skipped_irrelevant": 0,
        "skipped_duplicate": 0,
        "accepted": 0,
        "inserted": 0,
    }

    accepted_entries = []

    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num < args.start:
                continue
            if args.limit > 0 and stats["total_read"] >= args.limit:
                break

            stats["total_read"] += 1
            line = line.strip()
            if not line:
                stats["skipped_empty"] += 1
                continue

            entry = parse_jsonl_entry(line, line_num)
            if entry is None:
                stats["skipped_parse"] += 1
                continue

            content = extract_content(entry)
            if not content:
                stats["skipped_empty"] += 1
                continue

            wc = word_count(content)
            if wc < args.min_words:
                stats["skipped_short"] += 1
                continue
            if wc > args.max_words:
                stats["skipped_long"] += 1
                continue

            if not args.no_relevance_check and not is_glass_relevant(content):
                stats["skipped_irrelevant"] += 1
                continue

            ch = content_hash(content)
            if ch in existing_hashes:
                stats["skipped_duplicate"] += 1
                continue
            existing_hashes.add(ch)  # track within-file dupes too

            title = extract_title(entry, content)
            stats["accepted"] += 1

            accepted_entries.append({
                "title": title,
                "content": content,
                "source_type": args.source_type,
                "language": args.language,
                "metadata": {
                    "source_file": filepath.name,
                    "line_num": line_num,
                    "word_count": wc,
                    "category": entry.get("category", ""),
                    "difficulty": entry.get("difficulty", ""),
                },
            })

            # Progress
            if stats["total_read"] % 1000 == 0:
                print(f"  Processed {stats['total_read']:,} | Accepted: {stats['accepted']:,} | "
                      f"Rejected: {stats['total_read'] - stats['accepted']:,}")
                sys.stdout.flush()

    # ── Summary before insert ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FILTER RESULTS — {filepath.name}")
    print(f"{'='*60}")
    print(f"  Total read      : {stats['total_read']:,}")
    print(f"  Accepted        : {stats['accepted']:,} ({stats['accepted']/max(stats['total_read'],1)*100:.1f}%)")
    print(f"  Skipped parse   : {stats['skipped_parse']:,}")
    print(f"  Skipped empty   : {stats['skipped_empty']:,}")
    print(f"  Skipped short   : {stats['skipped_short']:,} (<{args.min_words} words)")
    print(f"  Skipped long    : {stats['skipped_long']:,} (>{args.max_words} words)")
    print(f"  Skipped irrelevant: {stats['skipped_irrelevant']:,}")
    print(f"  Skipped duplicate : {stats['skipped_duplicate']:,}")
    print(f"{'='*60}")
    sys.stdout.flush()

    if args.dry_run:
        print("\n  DRY RUN — nothing inserted.")
        # Show 3 sample accepted entries
        for i, e in enumerate(accepted_entries[:3]):
            print(f"\n  Sample {i+1}: [{e['title'][:60]}]")
            print(f"    {e['content'][:120]}...")
        return

    if not accepted_entries:
        print("  Nothing to insert.")
        return

    # ── Embed and insert ──────────────────────────────────────────────────────
    from ingestion.embedder import embed_texts
    from pgvector.psycopg2 import register_vector
    import psycopg2
    import numpy as np

    conn = psycopg2.connect(os.getenv(
        "DATABASE_URL",
        "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai"
    ))
    register_vector(conn)
    cur = conn.cursor()

    batch_size = args.batch_size
    total = len(accepted_entries)
    inserted = 0
    start_time = time.time()

    print(f"\n  Embedding and inserting {total:,} entries in batches of {batch_size}...")
    sys.stdout.flush()

    for i in range(0, total, batch_size):
        batch = accepted_entries[i:i+batch_size]
        texts = [e["content"] for e in batch]

        # Embed batch
        try:
            result = embed_texts(texts)
            embeddings = result["dense"]
        except Exception as e:
            print(f"  ERROR embedding batch {i}-{i+len(batch)}: {e}")
            continue

        # Insert batch
        for j, entry in enumerate(batch):
            try:
                cur.execute(
                    """
                    INSERT INTO documents (title, source_type, language, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entry["title"],
                        entry["source_type"],
                        entry["language"],
                        entry["content"],
                        json.dumps(entry["metadata"]),
                        embeddings[j].tolist(),
                    ),
                )
                inserted += 1
            except Exception as e:
                print(f"  ERROR inserting entry: {e}")
                conn.rollback()
                continue

        conn.commit()

        elapsed = time.time() - start_time
        rate = inserted / elapsed if elapsed > 0 else 0
        eta = (total - inserted) / rate if rate > 0 else 0
        print(f"  Batch {i//batch_size + 1}: inserted {inserted:,}/{total:,} "
              f"({inserted/total*100:.1f}%) | {rate:.1f} entries/sec | ETA: {eta:.0f}s")
        sys.stdout.flush()

    cur.close()
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Inserted: {inserted:,} / {total:,}")
    print(f"  Time: {elapsed:.1f}s ({inserted/elapsed:.1f} entries/sec)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
