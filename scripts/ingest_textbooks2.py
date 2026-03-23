"""
Glass Expert AI — Batch Ingest New Textbooks
==============================================
Ingests all PDFs from data/Textbooks 2/ into pgvector.
Also generates Q&A pairs from each textbook using GPT-4o-mini.

Usage:
    python scripts/ingest_textbooks2.py                    # ingest only
    python scripts/ingest_textbooks2.py --generate-qa      # ingest + generate Q&A pairs
    python scripts/ingest_textbooks2.py --generate-qa-only # Q&A only (already ingested)

Requires: postgres + redis running (docker compose up -d)
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from loguru import logger
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(REPO_ROOT / ".env")

TEXTBOOKS_DIR = REPO_ROOT / "data" / "Textbooks 2"
QA_OUTPUT = REPO_ROOT / "data" / "qa_pairs" / "generated" / "textbooks2_qa.jsonl"

# Q&A generation prompt
QA_SYSTEM = """You are a glass science professor creating exam questions from textbook passages.
For each passage, generate 3 high-quality Q&A pairs that test understanding of the material.

Rules:
- Questions should be specific and technical
- Answers should be 100-300 words with exact values from the text
- Include numerical data, formulas, and specific terminology
- Cover different aspects of the passage (theory, application, values)

Return JSON:
{
  "pairs": [
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."},
    {"question": "...", "answer": "..."}
  ]
}"""


def ingest_all():
    """Ingest all PDFs from Textbooks 2 directory."""
    from ingestion.ingest import ingest_document

    pdf_files = []
    for lang_dir in ["English", "Farsi"]:
        dir_path = TEXTBOOKS_DIR / lang_dir
        if dir_path.exists():
            for f in dir_path.glob("*.pdf"):
                language = "en" if lang_dir == "English" else "fa"
                pdf_files.append((f, language))

    logger.info(f"Found {len(pdf_files)} PDFs to ingest")

    results = []
    for pdf_path, language in pdf_files:
        logger.info(f"\nIngesting: {pdf_path.name} (lang={language})")
        try:
            stats = ingest_document(
                file_path=str(pdf_path),
                source_type="textbook",
                language=language,
                title=pdf_path.stem,
            )
            results.append({"file": pdf_path.name, "status": "ok", "chunks": stats.get("chunks", 0)})
            logger.info(f"  OK: {stats.get('chunks', 0)} chunks ingested")
        except Exception as e:
            results.append({"file": pdf_path.name, "status": "error", "error": str(e)})
            logger.error(f"  FAILED: {e}")

    logger.info(f"\n{'=' * 50}")
    logger.info("INGESTION COMPLETE")
    for r in results:
        status = "OK" if r["status"] == "ok" else "FAIL"
        logger.info(f"  [{status}] {r['file']}: {r.get('chunks', r.get('error', ''))}")
    return results


def generate_qa_from_textbooks():
    """Generate Q&A pairs from textbook content using GPT-4o-mini."""
    from ingestion.extractor import extract_file
    from ingestion.chunker import chunk_text

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    pdf_files = []
    for lang_dir in ["English", "Farsi"]:
        dir_path = TEXTBOOKS_DIR / lang_dir
        if dir_path.exists():
            for f in dir_path.glob("*.pdf"):
                language = "en" if lang_dir == "English" else "fa"
                pdf_files.append((f, language))

    logger.info(f"Generating Q&A from {len(pdf_files)} textbooks")

    all_qa = []
    # Resume from checkpoint
    if QA_OUTPUT.exists():
        with open(QA_OUTPUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_qa.append(json.loads(line))
        logger.info(f"  Resuming: {len(all_qa)} Q&A pairs already generated")
        done_sources = {p.get("source_file", "") for p in all_qa}
    else:
        done_sources = set()

    QA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(QA_OUTPUT, "a", encoding="utf-8") as save_file:
        for pdf_path, language in pdf_files:
            if pdf_path.name in done_sources:
                logger.info(f"  Skipping {pdf_path.name} (already done)")
                continue

            logger.info(f"\nProcessing: {pdf_path.name}")
            try:
                text, metadata = extract_file(str(pdf_path))
                if not text or len(text) < 100:
                    logger.warning(f"  No text extracted from {pdf_path.name}")
                    continue

                chunks = chunk_text(text, chunk_size=1500, chunk_overlap=100)
                logger.info(f"  {len(chunks)} chunks from {metadata.get('page_count', '?')} pages")

                # Generate Q&A for each chunk (max 50 chunks per PDF to control cost)
                for i, chunk in enumerate(chunks[:50]):
                    if len(chunk.split()) < 30:
                        continue

                    lang_instruction = ""
                    if language == "fa":
                        lang_instruction = "\n\nIMPORTANT: Generate questions and answers in Persian/Farsi."

                    user_prompt = f"""Textbook passage ({pdf_path.stem}, chunk {i+1}/{len(chunks)}):

{chunk[:3000]}
{lang_instruction}

Generate 3 Q&A pairs from this passage."""

                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": QA_SYSTEM},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.7,
                            max_tokens=2000,
                            response_format={"type": "json_object"},
                        )
                        data = json.loads(resp.choices[0].message.content)
                        pairs = data.get("pairs", [])

                        for p in pairs:
                            q = p.get("question", "")
                            a = p.get("answer", "")
                            if q and a and len(a.split()) >= 10:
                                qa_entry = {
                                    "question": q,
                                    "answer": a,
                                    "source_file": pdf_path.name,
                                    "language": language,
                                    "source_type": "textbook",
                                }
                                all_qa.append(qa_entry)
                                save_file.write(json.dumps(qa_entry, ensure_ascii=False) + "\n")
                                save_file.flush()

                        time.sleep(0.3)

                    except Exception as e:
                        logger.warning(f"  Q&A generation failed for chunk {i}: {e}")
                        continue

                    if (i + 1) % 10 == 0:
                        logger.info(f"  Chunk {i+1}/{min(len(chunks), 50)}: {len(all_qa)} total Q&A pairs")

            except Exception as e:
                logger.error(f"  Failed to process {pdf_path.name}: {e}")

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Q&A GENERATION COMPLETE: {len(all_qa)} total pairs")
    logger.info(f"Saved to: {QA_OUTPUT}")
    return all_qa


def main():
    parser = argparse.ArgumentParser(description="Ingest new textbooks")
    parser.add_argument("--generate-qa", action="store_true", help="Also generate Q&A pairs")
    parser.add_argument("--generate-qa-only", action="store_true", help="Only generate Q&A (skip ingestion)")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("Glass Expert AI — Batch Textbook Ingestion")
    logger.info(f"Source: {TEXTBOOKS_DIR}")
    logger.info("=" * 65)

    if not args.generate_qa_only:
        ingest_all()

    if args.generate_qa or args.generate_qa_only:
        generate_qa_from_textbooks()


if __name__ == "__main__":
    main()
