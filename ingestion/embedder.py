"""
Glass Expert AI — Embedding Module (Llama 8B Edition)
Supports:
  - BAAI/bge-m3 (multilingual, 1024-dim) — default, recommended
  - BAAI/bge-large-en-v1.5 (English-only, 1024-dim) — legacy
  - Any sentence-transformers compatible model

Set EMBEDDING_MODEL env var to switch models. Both bge-m3 and bge-large-en-v1.5
output 1024-dim vectors, so no DB schema change is needed when switching.
"""

import os
import threading
import numpy as np
from functools import lru_cache
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEVICE     = os.getenv("EMBEDDING_DEVICE", "cpu")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# BGE-M3 can use the FlagEmbedding library for best performance,
# but also works via sentence-transformers. We try FlagEmbedding first.
_USE_FLAG_EMBEDDING = MODEL_NAME == "BAAI/bge-m3"

# GPU concurrency limiter — prevents OOM when multiple requests embed simultaneously
_GPU_SEMAPHORE = threading.Semaphore(int(os.getenv("EMBEDDING_MAX_CONCURRENT", "4")))

# Query prefix for bge-m3 (per model card)
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── Lazy model loading (loads once on first use) ───────────────────────────────
@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the embedding model (true singleton via lru_cache)."""
    global _USE_FLAG_EMBEDDING

    logger.info(f"Loading embedding model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE.upper()}")
    logger.info("First run will download the model from HuggingFace...")

    if _USE_FLAG_EMBEDDING:
        try:
            from FlagEmbedding import BGEM3FlagModel
            model = BGEM3FlagModel(
                MODEL_NAME,
                use_fp16=(DEVICE != "cpu"),
            )
            logger.info("Loaded BGE-M3 via FlagEmbedding (multilingual, 1024-dim)")
            return model
        except ImportError:
            logger.info("FlagEmbedding not installed, falling back to sentence-transformers")
            _USE_FLAG_EMBEDDING = False

    # Fallback: sentence-transformers (works for both bge-m3 and bge-large-en-v1.5)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    logger.info(f"Loaded {MODEL_NAME} via sentence-transformers")
    return model


def warmup():
    """Pre-load model at startup. Call from app startup event."""
    logger.info("Warming up embedding model...")
    _load_model()
    logger.info("Embedding model ready.")


def embed_texts(texts: list, return_sparse: bool = False) -> dict:
    """
    Generate dense embeddings for a list of texts.

    Returns:
        dict with key 'dense' — numpy array of shape (N, 1024)
    """
    if not texts:
        return {"dense": np.array([]), "sparse": []}

    model = _load_model()
    logger.debug(f"Embedding {len(texts)} texts (batch_size={BATCH_SIZE})")

    with _GPU_SEMAPHORE:
        if _USE_FLAG_EMBEDDING:
            # BGE-M3 via FlagEmbedding — handles prefixing internally
            output = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                max_length=512,
            )
            dense_vecs = output["dense_vecs"]
            # Normalize (FlagEmbedding may not normalize by default)
            norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            dense_vecs = dense_vecs / norms
        else:
            # sentence-transformers path
            if "bge-large" in MODEL_NAME and "m3" not in MODEL_NAME:
                texts_to_encode = [_QUERY_PREFIX + t for t in texts]
            else:
                texts_to_encode = texts

            dense_vecs = model.encode(
                texts_to_encode,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    return {
        "dense": dense_vecs,
        "sparse": [],
    }


def embed_query(query: str) -> dict:
    """
    Embed a single query string (GPU semaphore protected).

    Returns:
        dict with 'dense' (1D numpy array of 1024 floats) and 'sparse'.
    """
    with _GPU_SEMAPHORE:
        model = _load_model()
        if _USE_FLAG_EMBEDDING:
            output = model.encode(
                [query],
                batch_size=1,
                max_length=512,
            )
            vec = output["dense_vecs"][0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        else:
            if "bge-large" in MODEL_NAME and "m3" not in MODEL_NAME:
                query = _QUERY_PREFIX + query
            vec = model.encode(
                [query],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

    return {
        "dense": vec,
        "sparse": {},
    }
