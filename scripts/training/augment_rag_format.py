"""
Glass Expert AI — RAG-Format Training Data Generator
=====================================================
Creates training examples that mirror actual RAG inference:
  Question + Retrieved Context Chunks → Answer with [Source N] citations

This teaches the LLM to:
  1. Ground answers in provided context (not hallucinate)
  2. Cite sources properly with [Source 1], [Source 2], etc.
  3. Synthesize information across multiple chunks
  4. Admit when context is insufficient

Pipeline:
  1. Load real chunks from pgvector (papers, textbooks, manuals, SOPs)
  2. For each Q&A pair, retrieve top-K relevant chunks via embedding similarity
  3. Send to GPT-4o-mini: "Given these chunks, answer this question with citations"
  4. Save in Qwen chat format with citation-grounded answers

Usage:
    python scripts/training/augment_rag_format.py                    # full run
    python scripts/training/augment_rag_format.py --limit 100        # test run
    python scripts/training/augment_rag_format.py --workers 30       # fewer workers
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import random
from pathlib import Path
from datetime import datetime

import psycopg2
import numpy as np
from openai import AsyncOpenAI

# ── Config ───────────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o-mini"
DB_URL = os.getenv("DATABASE_URL", "postgresql://glassai:glassai_secret@localhost:5432/glass_expert_ai")
TOP_K_CHUNKS = 5          # chunks to retrieve per question
SEMAPHORE_LIMIT = 50      # concurrent OpenAI calls
RPM_LIMIT = 2000          # rate limit
CHECKPOINT_EVERY = 500

INPUT_FILE = Path("data/processed/augmented_all_43k.jsonl")
OUTPUT_FILE = Path("data/processed/rag_format_training.jsonl")
CHECKPOINT_FILE = Path("data/processed/rag_format_checkpoint.json")
LOG_FILE = Path("rag_augment.log")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ── System prompt for RAG training ──────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are Glass Expert AI, a highly specialized assistant for glass science and engineering. You answer questions based ONLY on the provided source documents. Follow these rules strictly:

1. Base your answer entirely on the provided sources
2. Cite sources using [Source N] format (e.g., [Source 1], [Source 2])
3. If sources don't contain enough information, say so honestly
4. Provide technically precise, detailed answers
5. When multiple sources agree, synthesize them and cite all relevant ones"""

RAG_SYSTEM_PROMPT_FA = """شما Glass Expert AI هستید. فقط بر اساس منابع ارائه شده پاسخ دهید. از فرمت [Source N] برای ارجاع استفاده کنید. اگر منابع کافی نیستند، صادقانه بگویید."""

# ── GPT-4o-mini prompt for generating cited answers ─────────────────────────
GENERATION_PROMPT = """You are creating training data for a glass science RAG system. Given the question and retrieved source chunks below, write a high-quality answer that:

1. Uses ONLY information from the provided sources
2. Cites every claim with [Source N] (matching the source numbers provided)
3. Is technically precise and detailed (150-400 words)
4. Synthesizes information across sources when relevant
5. If sources are insufficient, acknowledge this but still extract what's useful

SOURCES:
{sources}

QUESTION: {question}

Write the answer with proper [Source N] citations:"""


class RateLimiter:
    """Token bucket rate limiter."""
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.interval = 60.0 / rpm
        self.last = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            wait = self.last + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self.last = time.monotonic()


def load_source_chunks(db_url: str) -> list[dict]:
    """Load all real source chunks (not QA pairs) from pgvector."""
    logger.info("Loading source chunks from database...")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, title, source_type, language, content, metadata
        FROM documents
        WHERE source_type != 'qa_pair'
        ORDER BY created_at
    """)

    chunks = []
    for row in cur.fetchall():
        chunks.append({
            "id": str(row[0]),
            "title": row[1] or "",
            "source_type": row[2] or "",
            "language": row[3] or "en",
            "content": row[4] or "",
            "metadata": row[5] or {},
        })

    conn.close()
    logger.info(f"  Loaded {len(chunks):,} source chunks (papers, textbooks, manuals, SOPs)")
    return chunks


def load_embeddings(db_url: str) -> tuple[list[str], np.ndarray]:
    """Load chunk embeddings for similarity search."""
    logger.info("Loading embeddings for similarity search...")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, embedding
        FROM documents
        WHERE source_type != 'qa_pair'
        AND embedding IS NOT NULL
    """)

    ids = []
    embeddings = []
    for row in cur.fetchall():
        ids.append(str(row[0]))
        # pgvector returns as string like '[0.1, 0.2, ...]'
        emb = row[1]
        if isinstance(emb, str):
            emb = json.loads(emb)
        embeddings.append(emb)

    conn.close()

    emb_matrix = np.array(embeddings, dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_matrix = emb_matrix / norms

    logger.info(f"  Loaded {len(ids):,} embeddings, dim={emb_matrix.shape[1]}")
    return ids, emb_matrix


def get_question_embedding(question: str, model_name: str = "BAAI/bge-large-en-v1.5") -> np.ndarray:
    """Get embedding for a question using the same model as the retriever."""
    from sentence_transformers import SentenceTransformer
    global _embed_model
    if '_embed_model' not in globals():
        logger.info(f"Loading embedding model: {model_name}")
        _embed_model = SentenceTransformer(model_name)
    emb = _embed_model.encode([question], normalize_embeddings=True)
    return emb[0]


def retrieve_chunks(
    question_emb: np.ndarray,
    chunk_ids: list[str],
    emb_matrix: np.ndarray,
    chunks_by_id: dict[str, dict],
    top_k: int = 5,
) -> list[dict]:
    """Retrieve top-K chunks by cosine similarity."""
    scores = emb_matrix @ question_emb  # cosine similarity (already normalized)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk_id = chunk_ids[idx]
        chunk = chunks_by_id.get(chunk_id)
        if chunk and scores[idx] > 0.3:  # minimum similarity threshold
            results.append({
                **chunk,
                "similarity": float(scores[idx]),
            })
    return results


def format_sources_for_prompt(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks as numbered sources for the prompt."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source_label = f"[Source {i}] ({chunk['source_type']}: {chunk['title'][:60]})"
        content = chunk['content'][:800]  # truncate long chunks
        parts.append(f"{source_label}\n{content}")
    return "\n\n".join(parts)


def format_sources_for_training(retrieved_chunks: list[dict]) -> str:
    """Format sources as they appear in the training user message."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        title = chunk['title'][:60]
        stype = chunk['source_type']
        content = chunk['content'][:800]
        parts.append(f"[Source {i}] ({stype}: {title})\n{content}")
    return "\n\n".join(parts)


async def generate_cited_answer(
    client: AsyncOpenAI,
    question: str,
    retrieved_chunks: list[dict],
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
) -> str | None:
    """Generate a citation-grounded answer using GPT-4o-mini."""
    sources_text = format_sources_for_prompt(retrieved_chunks)

    prompt = GENERATION_PROMPT.format(
        sources=sources_text,
        question=question,
    )

    async with semaphore:
        await rate_limiter.acquire()
        try:
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a glass science expert creating training data. Write detailed, citation-grounded answers."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"  GPT error: {e}")
            return None


async def process_batch(
    batch: list[dict],
    batch_start_idx: int,
    client: AsyncOpenAI,
    chunk_ids: list[str],
    emb_matrix: np.ndarray,
    chunks_by_id: dict[str, dict],
    semaphore: asyncio.Semaphore,
    rate_limiter: RateLimiter,
    embed_model,
) -> list[dict]:
    """Process a batch of questions: retrieve chunks, generate cited answers."""

    results = []

    # Pre-compute question embeddings (CPU, batched)
    questions = [item["question"] for item in batch]
    q_embeddings = embed_model.encode(questions, normalize_embeddings=True, show_progress_bar=False)

    # Create async tasks for GPT calls
    async def process_one(item: dict, q_emb: np.ndarray, idx: int) -> dict | None:
        question = item["question"]

        # Retrieve relevant chunks
        retrieved = retrieve_chunks(q_emb, chunk_ids, emb_matrix, chunks_by_id, TOP_K_CHUNKS)

        if len(retrieved) < 2:
            return None  # skip if not enough relevant context

        # Generate cited answer
        answer = await generate_cited_answer(
            client, question, retrieved, semaphore, rate_limiter
        )

        if not answer or "[Source" not in answer:
            return None  # skip if no citations generated

        # Detect language
        is_farsi = any('\u0600' <= c <= '\u06FF' for c in question[:50])
        sys_prompt = RAG_SYSTEM_PROMPT_FA if is_farsi else RAG_SYSTEM_PROMPT

        # Format as Qwen chat training example
        sources_text = format_sources_for_training(retrieved)
        user_message = f"""Based on the following sources, answer the question.

{sources_text}

Question: {question}"""

        return {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer},
            ],
            "type": "rag_grounded",
            "num_sources": len(retrieved),
            "avg_similarity": round(sum(c["similarity"] for c in retrieved) / len(retrieved), 3),
        }

    tasks = [
        process_one(item, q_embeddings[i], batch_start_idx + i)
        for i, item in enumerate(batch)
    ]

    completed = await asyncio.gather(*tasks)
    results = [r for r in completed if r is not None]
    return results


async def main():
    parser = argparse.ArgumentParser(description="Generate RAG-format training data")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    parser.add_argument("--workers", type=int, default=SEMAPHORE_LIMIT, help="Concurrent workers")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("Glass Expert AI -- RAG-Format Training Data Generator")
    logger.info(f"  Model: {OPENAI_MODEL}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Top-K chunks: {TOP_K_CHUNKS}")
    logger.info("=" * 65)

    # ── Load source chunks from DB ──────────────────────────────────────────
    chunks = load_source_chunks(DB_URL)
    chunks_by_id = {c["id"]: c for c in chunks}

    # ── Load embeddings ─────────────────────────────────────────────────────
    chunk_ids, emb_matrix = load_embeddings(DB_URL)
    logger.info(f"  Embedding matrix: {emb_matrix.shape}")

    # ── Load embedding model ────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: BAAI/bge-large-en-v1.5")
    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # ── Load questions from augmented data ──────────────────────────────────
    logger.info(f"Loading questions from {INPUT_FILE}")
    questions = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            q = d.get("question", "").strip()
            if q:
                questions.append({"question": q})

    random.seed(42)
    random.shuffle(questions)

    if args.limit > 0:
        questions = questions[:args.limit]

    logger.info(f"  Total questions: {len(questions):,}")

    # ── Resume from checkpoint ──────────────────────────────────────────────
    start_idx = 0
    if args.resume and CHECKPOINT_FILE.exists():
        cp = json.loads(CHECKPOINT_FILE.read_text())
        start_idx = cp.get("processed", 0)
        logger.info(f"  Resuming from checkpoint: {start_idx:,}")

    # ── OpenAI client ───────────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set!")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.workers)
    rate_limiter = RateLimiter(RPM_LIMIT)

    # ── Process in batches ──────────────────────────────────────────────────
    total_generated = 0
    total_skipped = 0
    batch_size = args.batch_size
    start_time = time.time()

    # Open output file in append mode
    mode = "a" if args.resume and start_idx > 0 else "w"
    out_f = open(OUTPUT_FILE, mode, encoding="utf-8")

    try:
        for batch_start in range(start_idx, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch = questions[batch_start:batch_end]

            results = await process_batch(
                batch, batch_start, client,
                chunk_ids, emb_matrix, chunks_by_id,
                semaphore, rate_limiter, embed_model,
            )

            # Write results
            for r in results:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()

            total_generated += len(results)
            total_skipped += len(batch) - len(results)

            # Progress
            elapsed = time.time() - start_time
            rate = (batch_end - start_idx) / elapsed if elapsed > 0 else 0
            eta = (len(questions) - batch_end) / rate if rate > 0 else 0

            logger.info(
                f"  [{batch_end:,}/{len(questions):,}] "
                f"generated={total_generated:,} skipped={total_skipped:,} "
                f"rate={rate:.0f}/s ETA={eta/60:.0f}min"
            )

            # Checkpoint
            if batch_end % CHECKPOINT_EVERY < batch_size:
                CHECKPOINT_FILE.write_text(json.dumps({
                    "processed": batch_end,
                    "generated": total_generated,
                    "skipped": total_skipped,
                    "timestamp": datetime.now().isoformat(),
                }))

    finally:
        out_f.close()

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("=" * 65)
    logger.info("RAG-FORMAT AUGMENTATION COMPLETE")
    logger.info(f"  Total processed: {len(questions):,}")
    logger.info(f"  Generated: {total_generated:,}")
    logger.info(f"  Skipped (low similarity): {total_skipped:,}")
    logger.info(f"  Time: {elapsed/60:.1f} min")
    logger.info(f"  Output: {OUTPUT_FILE}")
    logger.info(f"  Cost estimate: ~${total_generated * 0.0003:.2f}")
    logger.info("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
