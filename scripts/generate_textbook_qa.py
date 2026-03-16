"""
Glass Expert AI — Generate Q&A Pairs from Textbooks 2
======================================================
Extracts text from all PDFs in data/Textbooks 2/, chunks them,
and uses GPT-4o-mini to generate high-quality Q&A training pairs.

Saves incrementally to data/qa_pairs/generated/textbooks2_qa.jsonl.
Crash-safe: skips already-processed source+chunk combos on restart.

Usage:
    python scripts/generate_textbook_qa.py
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(REPO_ROOT / ".env")

# Use absolute path to Textbooks 2 (in main repo, not worktree)
TEXTBOOKS_DIR = Path("C:/Users/zainm/glass-ai-data/glass-expert-ai/data/Textbooks 2")
QA_OUTPUT = REPO_ROOT / "data" / "qa_pairs" / "generated" / "textbooks2_qa.jsonl"

# ── Q&A generation prompts ────────────────────────────────────────────────────

QA_SYSTEM_EN = """You are a glass science professor creating exam-quality Q&A pairs from textbook passages.

For EACH passage, generate 3-5 Q&A pairs that:
1. Test DIFFERENT aspects (theory, application, numerical values, process steps)
2. Range from intermediate to advanced difficulty
3. Include SPECIFIC data: temperatures, compositions (e.g., 72% SiO2), property values, standards
4. Have answers of 150-300 words with precise technical detail
5. Are self-contained (answerable without seeing the passage)

Question types to include:
- "What is..." / "Define..." (concept questions)
- "How does..." / "Explain the process..." (mechanism questions)
- "What are the effects of..." (cause-effect questions)
- "Compare..." / "What is the difference between..." (comparison questions)
- "Calculate..." / "What value..." (quantitative questions, when data available)

Return ONLY valid JSON:
{
  "pairs": [
    {"question": "...", "answer": "...", "difficulty": "intermediate|advanced", "category": "..."},
    ...
  ]
}"""

QA_SYSTEM_FA = """شما یک استاد علم شیشه هستید که سوالات امتحانی با کیفیت بالا از متون درسی تهیه می‌کنید.

برای هر متن، ۳ تا ۵ جفت سوال و جواب تولید کنید که:
۱. جنبه‌های مختلف را پوشش دهند (تئوری، کاربرد، مقادیر عددی، مراحل فرآیند)
۲. از سطح متوسط تا پیشرفته باشند
۳. شامل داده‌های خاص: دماها، ترکیبات، مقادیر خواص، استانداردها
۴. پاسخ‌ها ۱۵۰ تا ۳۰۰ کلمه با جزئیات فنی دقیق باشند
۵. مستقل باشند (بدون دیدن متن قابل پاسخ باشند)

فقط JSON معتبر برگردانید:
{
  "pairs": [
    {"question": "...", "answer": "...", "difficulty": "intermediate|advanced", "category": "..."},
    ...
  ]
}"""


def chunk_text_simple(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Simple token-approximate chunker (no external deps needed)."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 30:  # skip tiny chunks
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Find all PDFs
    pdf_files = []
    for lang_dir in ["English", "Farsi"]:
        d = TEXTBOOKS_DIR / lang_dir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.pdf")):
            language = "en" if lang_dir == "English" else "fa"
            pdf_files.append((f, language))

    print(f"{'=' * 65}")
    print(f"  Glass Expert AI - Textbook Q&A Generation")
    print(f"  PDFs: {len(pdf_files)}  |  Output: {QA_OUTPUT}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 65}")

    # Load already-processed chunks for resume
    QA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    existing_count = 0
    if QA_OUTPUT.exists():
        with open(QA_OUTPUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        key = f"{entry.get('source_file', '')}|{entry.get('chunk_idx', '')}"
                        done_keys.add(key)
                        existing_count += 1
                    except json.JSONDecodeError:
                        pass
        print(f"  Resuming: {existing_count} Q&A pairs already generated")

    # Extract text using PyMuPDF directly (avoid extractor return format issues)
    import fitz

    total_pairs = existing_count
    total_chunks_processed = 0

    with open(QA_OUTPUT, "a", encoding="utf-8") as save_file:
        for pdf_idx, (pdf_path, language) in enumerate(pdf_files):
            print(f"\n[{pdf_idx + 1}/{len(pdf_files)}] {pdf_path.name} (lang={language})")

            try:
                doc = fitz.open(str(pdf_path))
                pages_text = []
                for page in doc:
                    text = page.get_text("text").strip()
                    if text:
                        pages_text.append(text)
                doc.close()

                full_text = "\n\n".join(pages_text)
                if len(full_text) < 100:
                    print(f"  Skipping: too little text ({len(full_text)} chars)")
                    continue

                print(f"  Extracted: {len(full_text):,} chars from {len(pages_text)} pages")

                # Chunk the text
                chunks = chunk_text_simple(full_text, chunk_size=400, overlap=50)
                max_chunks = 60  # cap per PDF to control cost
                chunks = chunks[:max_chunks]
                print(f"  Chunks: {len(chunks)} (max {max_chunks})")

                chunk_pairs = 0
                for i, chunk in enumerate(chunks):
                    # Skip if already done
                    key = f"{pdf_path.name}|{i}"
                    if key in done_keys:
                        continue

                    # Skip very short chunks
                    if len(chunk.split()) < 40:
                        continue

                    system = QA_SYSTEM_FA if language == "fa" else QA_SYSTEM_EN
                    lang_note = "\n\nIMPORTANT: Generate ALL questions and answers in Persian/Farsi." if language == "fa" else ""

                    user_prompt = f"""Textbook: {pdf_path.stem}
Passage ({i + 1}/{len(chunks)}):

{chunk[:4000]}
{lang_note}

Generate Q&A pairs from this passage."""

                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.7,
                            max_tokens=3000,
                            response_format={"type": "json_object"},
                        )

                        data = json.loads(resp.choices[0].message.content)
                        pairs = data.get("pairs", [])

                        for p in pairs:
                            q = p.get("question", "")
                            a = p.get("answer", "")
                            if q and a and len(a.split()) >= 15:
                                entry = {
                                    "question": q,
                                    "answer": a,
                                    "difficulty": p.get("difficulty", "intermediate"),
                                    "category": p.get("category", "glass_science"),
                                    "source_file": pdf_path.name,
                                    "chunk_idx": i,
                                    "language": language,
                                    "source_type": "textbook",
                                    "generated_by": "gpt-4o-mini",
                                }
                                save_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                                total_pairs += 1
                                chunk_pairs += 1

                        save_file.flush()
                        time.sleep(0.3)  # rate limit

                    except Exception as e:
                        print(f"  Chunk {i} API error: {e}")
                        time.sleep(2)
                        continue

                    total_chunks_processed += 1
                    if (i + 1) % 10 == 0:
                        print(f"  Progress: chunk {i + 1}/{len(chunks)}, {chunk_pairs} pairs from this PDF, {total_pairs} total")

                print(f"  Done: +{chunk_pairs} Q&A pairs from {pdf_path.name}")

            except Exception as e:
                print(f"  FAILED: {e}")
                continue

    print(f"\n{'=' * 65}")
    print(f"  Q&A GENERATION COMPLETE")
    print(f"  Total Q&A pairs: {total_pairs}")
    print(f"  Chunks processed: {total_chunks_processed}")
    print(f"  Output: {QA_OUTPUT}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
