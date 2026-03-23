"""
Glass Expert AI — Generate RAG-format Training Data
====================================================
Creates training examples that match the EXACT production prompt format:
  [system]: RAG system prompt with citation rules
  [user]: KNOWLEDGE BASE CONTEXT:\n[Source 1]...\n\nQUESTION: ...
  [assistant]: Answer with [Source N] citations

This teaches the model to:
1. Read and use provided context
2. Cite sources with [Source N]
3. Synthesize info from multiple sources
4. Say "the context does not specify" when info is missing
5. Answer in the correct language (EN or FA)

Sources:
  - Original 82K Q&A pairs -> reformatted with synthetic context
  - Document chunks -> self-contained Q&A from content
  - SciGlass compositions -> property lookup with context

Usage:
    python scripts/generate_rag_training.py
"""
import os
import sys
import json
import re
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", level="INFO")

DATA_DIR = Path("models/qwen14b-glass-expert/data")

# ── Production system prompts (must match retrieval/llm.py exactly) ───────────
SYSTEM_PROMPT_EN = """You are Glass Expert AI, a highly specialized assistant for glass scientists and manufacturing engineers with PhD-level expertise.

CRITICAL RULES:
- Answer ONLY using information from the KNOWLEDGE BASE CONTEXT provided below.
- Do NOT fabricate values, compositions, temperatures, or any data not present in the context.
- If the context lacks specific information, state "the provided context does not specify..." rather than guessing.
- Extract and quote exact numerical values, ranges, and units directly from the context.
- When multiple sources provide data on the same topic, SYNTHESIZE them — combine complementary details and note any conflicts between sources.
- Every numerical claim (temperature, composition, property value) MUST be traceable to a specific [Source N].

ANSWER STRUCTURE:
1. Lead with a direct, concise answer to the question (1-2 sentences).
2. Follow with detailed technical explanation using numbered points.
3. Include specific compositions (e.g., 72% SiO2, 14% Na2O) and property values FROM the context.
4. For processes, list ALL stages with their specific temperature ranges and conditions.
5. Reference named scientists, equations (e.g., Abbe number V=(n_d-1)/(n_F-n_C)), and standards (ISO, ASTM) when they appear in the context.
6. Use precise technical terms: network formers, network modifiers, bridging oxygen (BO), non-bridging oxygen (NBO), coordination number, fining agents, devitrification, etc.
7. Cite sources as [Source 1], [Source 2], etc. for every key fact.
8. End with a brief summary if the answer covers multiple aspects.
9. Keep answers focused, quantitative, and specific — never give vague generalizations."""

SYSTEM_PROMPT_FA = """شما Glass Expert AI هستید، یک دستیار تخصصی با تخصص سطح دکترا برای دانشمندان شیشه و مهندسان تولید.

قواعد حیاتی:
- فقط بر اساس متن پایگاه دانش ارائه شده پاسخ دهید — هرگز اطلاعات جعلی ارائه ندهید
- مقادیر، فرمول‌ها و ترکیبات دقیق را مستقیماً از متن استخراج و نقل کنید
- اگر متن اطلاعات کافی ندارد، بگویید «متن ارائه شده این اطلاعات را مشخص نمی‌کند» — حدس نزنید

قالب‌بندی:
1. با تعریف یا پاسخ مستقیم شروع کنید
2. از لیست شماره‌دار برای فرآیندها، قوانین و روش‌ها استفاده کنید
3. ترکیبات خاص (مثلاً ۷۲٪ SiO2) و مقادیر خواص را از متن ذکر کنید
4. نام سند منبع را در کروشه ذکر کنید، مثلاً [منبع ۱]
5. اگر منابع مختلف اطلاعات متناقضی دارند، تناقض را ذکر کنید
6. مقادیر عددی را با واحد و شرایط (دما، فشار، ترکیب) ذکر کنید"""


def format_context_block(sources: list[dict]) -> str:
    """Format sources into production context block format."""
    parts = ["KNOWLEDGE BASE CONTEXT:", "=" * 50]
    for i, src in enumerate(sources, 1):
        title = src.get("title", "Unknown")
        stype = src.get("source_type", "reference")
        lang = src.get("language", "en").upper()
        sim = src.get("similarity", 0.85)
        parts.append(
            f"\n[Source {i}] {title} "
            f"(Type: {stype} | Language: {lang} | Relevance: {sim:.0%})"
        )
        parts.append(src["content"])
        parts.append("-" * 40)
    return "\n".join(parts)


def format_user_message_en(context_block: str, question: str) -> str:
    return f"""{context_block}

QUESTION: {question}

Provide a precise, technical answer based strictly on the knowledge base context above. Reference sources by their [Source N] numbers."""


def format_user_message_fa(context_block: str, question: str) -> str:
    return f"""{context_block}

سوال: {question}

لطفاً یک پاسخ دقیق و فنی بر اساس متن پایگاه دانش بالا ارائه دهید. منابع را با شماره [منبع N] ارجاع دهید.

IMPORTANT: You MUST answer entirely in Persian/Farsi. Do NOT answer in English."""


# ── Transform existing Q&A pairs to RAG format ──────────────────────────────

def transform_original_qa(line: str) -> dict | None:
    """Transform an original Q&A pair into RAG format.
    The original answer becomes the context (as if retrieved), and
    we wrap it in the production format with source citations."""
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    messages = data.get("messages", [])
    if len(messages) < 3:
        return None

    system_msg = messages[0]["content"]
    question = messages[1]["content"]
    original_answer = messages[2]["content"]

    if len(original_answer) < 50:
        return None

    # Detect language from system prompt
    is_farsi = "شما" in system_msg or "فارسی" in system_msg

    # The original answer becomes the "retrieved context"
    # We simulate 1-2 source chunks from the answer content
    answer_parts = original_answer.split("\n\n")

    sources = []
    if len(answer_parts) >= 2:
        # Split into 2 sources for multi-source training
        mid = len(answer_parts) // 2
        sources.append({
            "title": _generate_title(question),
            "source_type": random.choice(["paper", "textbook", "manual"]),
            "language": "fa" if is_farsi else "en",
            "similarity": round(random.uniform(0.78, 0.95), 2),
            "content": "\n\n".join(answer_parts[:mid]),
        })
        sources.append({
            "title": _generate_title_2(question),
            "source_type": random.choice(["paper", "textbook"]),
            "language": "fa" if is_farsi else "en",
            "similarity": round(random.uniform(0.65, 0.85), 2),
            "content": "\n\n".join(answer_parts[mid:]),
        })
    else:
        sources.append({
            "title": _generate_title(question),
            "source_type": random.choice(["paper", "textbook", "manual"]),
            "language": "fa" if is_farsi else "en",
            "similarity": round(random.uniform(0.78, 0.95), 2),
            "content": original_answer,
        })

    context_block = format_context_block(sources)

    # Build the answer WITH citations
    cited_answer = _add_citations(original_answer, len(sources))

    if is_farsi:
        user_msg = format_user_message_fa(context_block, question)
        sys_prompt = SYSTEM_PROMPT_FA
    else:
        user_msg = format_user_message_en(context_block, question)
        sys_prompt = SYSTEM_PROMPT_EN

    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": cited_answer},
        ]
    }


def _generate_title(question: str) -> str:
    """Generate a plausible document title from a question."""
    # Extract key nouns
    q = question.lower()
    if "composition" in q or "sio2" in q.lower() or "glass id" in q:
        return random.choice([
            "Glass Composition Database - SciGlass",
            "Oxide Glass Compositions and Properties",
            "Glass Science Handbook - Compositions",
        ])
    elif "thermal" in q or "temperature" in q or "tg" in q:
        return random.choice([
            "Thermal Properties of Glass Materials",
            "Glass Transition and Crystallization Studies",
            "Thermodynamic Properties of Silicate Glasses",
        ])
    elif "optical" in q or "refractive" in q:
        return random.choice([
            "Optical Properties of Glasses",
            "Refractive Index Studies in Glass Systems",
        ])
    elif "defect" in q or "bubble" in q or "stone" in q:
        return random.choice([
            "Glass Defects and Quality Control",
            "Manufacturing Defects in Glass Production",
        ])
    else:
        return random.choice([
            "Fundamentals of Glass Science and Technology",
            "Glass Engineering Principles",
            "Advanced Glass Materials Research",
            "Glass Science and Technology Handbook",
            "Modern Glass Manufacturing",
        ])


def _generate_title_2(question: str) -> str:
    """Generate a second plausible title."""
    return random.choice([
        "Recent Advances in Glass Research",
        "Glass Properties and Applications",
        "Materials Science of Glass",
        "Glass Technology Review",
        "Practical Glass Manufacturing Guide",
    ])


def _add_citations(answer: str, num_sources: int) -> str:
    """Add [Source N] citations to an answer."""
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if not sentences:
        return answer

    cited_sentences = []
    for i, sent in enumerate(sentences):
        # Add citation to sentences with data (numbers, percentages, temperatures)
        has_data = bool(re.search(r'\d+[\.,]?\d*\s*[%°]|mol%|wt%|\d+\s*°C|\d+\s*K|\d+\s*GPa|\d+\s*MPa', sent))
        if has_data or i == 0 or (i == len(sentences) - 1 and len(sentences) > 2):
            source_num = random.randint(1, num_sources)
            if f"[Source" not in sent:
                sent = sent.rstrip('.') + f" [Source {source_num}]."
        cited_sentences.append(sent)

    return " ".join(cited_sentences)


# ── Transform document chunks to RAG format ────────────────────────────────

def transform_chunk_qa(chunk_line: str) -> dict | None:
    """Transform a chunk Q&A into proper RAG format."""
    try:
        data = json.loads(chunk_line)
    except json.JSONDecodeError:
        return None

    messages = data.get("messages", [])
    if len(messages) < 3:
        return None

    question = messages[1]["content"]
    answer = messages[2]["content"]

    if len(answer) < 100:
        return None

    # Extract title from the answer (usually starts with 'Based on the ...')
    title_match = re.search(r'"([^"]+)"', answer[:200])
    title = title_match.group(1) if title_match else "Glass Science Reference"

    # Clean up the content: remove the "Based on the..." prefix
    content = re.sub(r'^Based on the \w+ "[^"]*":\s*', '', answer).strip()
    if len(content) < 50:
        return None

    is_farsi = "شما" in messages[0]["content"]

    # Create the source context
    sources = [{
        "title": title,
        "source_type": random.choice(["paper", "textbook", "manual"]),
        "language": "fa" if is_farsi else "en",
        "similarity": round(random.uniform(0.75, 0.93), 2),
        "content": content[:1500],
    }]

    context_block = format_context_block(sources)

    # Generate a proper answer WITH citations referencing the context
    cited_answer = f"Based on the provided context from \"{title}\" [Source 1]:\n\n"
    # Take key sentences from content and synthesize
    content_sentences = re.split(r'(?<=[.!?])\s+', content[:800])
    if len(content_sentences) > 3:
        cited_answer += " ".join(content_sentences[:3]) + " [Source 1]."
        if len(content_sentences) > 5:
            cited_answer += "\n\n" + " ".join(content_sentences[3:6]) + " [Source 1]."
    else:
        cited_answer += content[:500] + " [Source 1]."

    if is_farsi:
        # Fix Farsi question if it's garbage
        if len(question) > 10:
            user_msg = format_user_message_fa(context_block, question)
            sys_prompt = SYSTEM_PROMPT_FA
        else:
            return None
    else:
        user_msg = format_user_message_en(context_block, question)
        sys_prompt = SYSTEM_PROMPT_EN

    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": cited_answer},
        ]
    }


# ── SciGlass to RAG format ─────────────────────────────────────────────────

def load_property_names():
    prop_path = Path("data/csv/LISTPROP.csv")
    if prop_path.exists():
        df = pd.read_csv(str(prop_path), encoding="utf-8", on_bad_lines="skip")
        mapping = {}
        for _, row in df.iterrows():
            fld = str(row.get("FLDNAM", "")).strip().lower()
            name = str(row.get("NAME", "")).strip()
            unit = str(row.get("Unit", "")).strip()
            if fld and name:
                mapping[fld] = f"{name} ({unit})" if unit and unit != "nan" else name
        return mapping
    return {}


SCIGLASS_QUESTIONS = [
    "What is the composition and properties of glass #{glass_id}?",
    "Describe the glass #{glass_id} from the SciGlass database.",
    "What are the measured properties of glass #{glass_id}?",
    "Tell me about the composition of glass #{glass_id}.",
    "What glass system does #{glass_id} belong to?",
]


def sciglass_row_to_rag(row, prop_names) -> dict | None:
    """Convert a SciGlass row to RAG-format training example."""
    glass_id = row.get("glass_id", "unknown")
    comps = []
    props = []

    for col, val in row.items():
        if pd.isna(val) or str(val).strip() == "" or col == "glass_id":
            continue
        val_str = str(val).strip()

        if col.startswith("comp_"):
            component = col[5:].upper().replace("_", "")
            try:
                fval = float(val_str)
                if fval > 0:
                    comps.append((component, fval))
            except ValueError:
                pass
        elif col.startswith("prop_") or col.startswith("t_") or col.startswith("d_"):
            readable = prop_names.get(col.lower(), col)
            try:
                fval = float(val_str)
                props.append((readable, f"{fval:.4g}"))
            except ValueError:
                props.append((readable, val_str))

    if not comps and not props:
        return None

    # Build natural language context (what the retriever would return)
    context_lines = [f"Glass #{glass_id} from the SciGlass database:"]
    if comps:
        comp_str = ", ".join(f"{c}: {v:.2f} mol%" for c, v in sorted(comps, key=lambda x: -x[1]))
        context_lines.append(f"Composition: {comp_str}")

        # Identify glass system
        major = [c for c, v in comps if v > 5]
        if major:
            system = "-".join(sorted(major[:5]))
            context_lines.append(f"Glass system: {system}")

    if props:
        prop_str = ", ".join(f"{n}: {v}" for n, v in props[:15])
        context_lines.append(f"Measured properties: {prop_str}")

    content = "\n".join(context_lines)

    # Build source
    sources = [{
        "title": f"SciGlass Database - Glass #{glass_id}",
        "source_type": "standard",
        "language": "EN",
        "similarity": round(random.uniform(0.82, 0.97), 2),
        "content": content,
    }]

    context_block = format_context_block(sources)
    question = random.choice(SCIGLASS_QUESTIONS).format(glass_id=glass_id)
    user_msg = format_user_message_en(context_block, question)

    # Build proper cited answer
    answer_parts = [f"Glass #{glass_id} is documented in the SciGlass database [Source 1]."]

    if comps:
        top_comps = sorted(comps, key=lambda x: -x[1])[:5]
        comp_desc = ", ".join(f"{c} at {v:.2f} mol%" for c, v in top_comps)
        answer_parts.append(f"\nComposition: The primary components are {comp_desc} [Source 1].")

        major = [c for c, v in comps if v > 5]
        if len(major) >= 2:
            answer_parts.append(f"This places it in the {'-'.join(sorted(major[:5]))} glass system.")

    if props:
        prop_desc = ", ".join(f"{n}: {v}" for n, v in props[:10])
        answer_parts.append(f"\nMeasured properties: {prop_desc} [Source 1].")

    answer = "\n".join(answer_parts)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_EN},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ]
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sciglass-sample", type=int, default=50000,
                        help="Max SciGlass examples to generate")
    args = parser.parse_args()

    random.seed(42)

    # ── 1. Transform original Q&A pairs ──────────────────────────────────────
    logger.info("Processing original Q&A pairs...")
    original_path = DATA_DIR / "combined_train.jsonl"
    rag_original = []
    skipped = 0
    with open(original_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = transform_original_qa(line)
            if result:
                rag_original.append(result)
            else:
                skipped += 1
    logger.info(f"  Original Q&A: {len(rag_original):,} converted, {skipped:,} skipped")

    # ── 2. Transform document chunk Q&A ──────────────────────────────────────
    logger.info("Processing document chunk Q&A...")
    chunks_path = DATA_DIR / "chunks_train.jsonl"
    rag_chunks = []
    skipped = 0
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result = transform_chunk_qa(line)
                if result:
                    rag_chunks.append(result)
                else:
                    skipped += 1
        logger.info(f"  Chunk Q&A: {len(rag_chunks):,} converted, {skipped:,} skipped")

    # ── 3. Generate SciGlass RAG examples ────────────────────────────────────
    logger.info("Processing SciGlass CSV...")
    prop_names = load_property_names()
    logger.info(f"  Loaded {len(prop_names)} property name mappings")

    csv_path = Path("data/csv/glass_data.csv")
    rag_sciglass = []
    row_count = 0

    chunk_iter = pd.read_csv(
        str(csv_path), encoding="utf-8", on_bad_lines="skip",
        chunksize=10000, low_memory=False
    )

    for df_chunk in chunk_iter:
        for _, row in df_chunk.iterrows():
            row_count += 1
            if len(rag_sciglass) >= args.sciglass_sample:
                break
            result = sciglass_row_to_rag(row, prop_names)
            if result:
                rag_sciglass.append(result)
            if len(rag_sciglass) % 10000 == 0 and len(rag_sciglass) > 0:
                logger.info(f"  SciGlass: {len(rag_sciglass):,} examples from {row_count:,} rows...")
        if len(rag_sciglass) >= args.sciglass_sample:
            break

    # Downsample if needed
    if len(rag_sciglass) > args.sciglass_sample:
        rag_sciglass = random.sample(rag_sciglass, args.sciglass_sample)

    logger.info(f"  SciGlass: {len(rag_sciglass):,} RAG examples generated")

    # ── 4. Combine and shuffle ───────────────────────────────────────────────
    all_data = rag_original + rag_chunks + rag_sciglass
    random.shuffle(all_data)

    logger.info(f"\nTotal RAG training examples: {len(all_data):,}")
    logger.info(f"  Original Q&A: {len(rag_original):,} ({len(rag_original)/len(all_data)*100:.1f}%)")
    logger.info(f"  Doc chunks:   {len(rag_chunks):,} ({len(rag_chunks)/len(all_data)*100:.1f}%)")
    logger.info(f"  SciGlass:     {len(rag_sciglass):,} ({len(rag_sciglass)/len(all_data)*100:.1f}%)")

    # ── 5. Train/val split ───────────────────────────────────────────────────
    val_size = max(int(len(all_data) * 0.02), 1000)
    val_indices = set(random.sample(range(len(all_data)), val_size))

    train_path = DATA_DIR / "rag_train.jsonl"
    val_path = DATA_DIR / "rag_val.jsonl"

    train_count = 0
    val_count = 0
    with open(train_path, "w", encoding="utf-8") as tf, \
         open(val_path, "w", encoding="utf-8") as vf:
        for i, example in enumerate(all_data):
            line = json.dumps(example, ensure_ascii=False) + "\n"
            if i in val_indices:
                vf.write(line)
                val_count += 1
            else:
                tf.write(line)
                train_count += 1

    logger.info(f"\nFinal RAG-format training data:")
    logger.info(f"  Train: {train_count:,} -> {train_path}")
    logger.info(f"  Val:   {val_count:,} -> {val_path}")

    # Print a sample for verification
    sample = random.choice(all_data)
    logger.info(f"\n{'='*60}")
    logger.info(f"SAMPLE TRAINING EXAMPLE:")
    logger.info(f"{'='*60}")
    for msg in sample["messages"]:
        preview = msg["content"][:200].encode("ascii", "replace").decode()
        logger.info(f"  [{msg['role'].upper()}]: {preview}...")


if __name__ == "__main__":
    main()
