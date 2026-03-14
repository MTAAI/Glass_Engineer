"""
Glass Expert AI — Embedding Module
Supports:
  - BAAI/bge-m3 (multilingual, 1024-dim) — default, recommended
  - BAAI/bge-large-en-v1.5 (English-only, 1024-dim) — legacy
  - Any sentence-transformers compatible model

Set EMBEDDING_MODEL env var to switch models. Both bge-m3 and bge-large-en-v1.5
output 1024-dim vectors, so no DB schema change is needed when switching.
"""

import os
import numpy as np
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

# ── Lazy model loading (loads once on first use) ───────────────────────────────
_model = None


def _get_model():
    """Load and cache the embedding model (singleton pattern)."""
    global _model, _USE_FLAG_EMBEDDING
    if _model is not None:
        return _model

    logger.info(f"Loading embedding model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE.upper()}")
    logger.info("First run will download the model from HuggingFace...")

    if _USE_FLAG_EMBEDDING:
        try:
            from FlagEmbedding import BGEM3FlagModel
            _model = BGEM3FlagModel(
                MODEL_NAME,
                use_fp16=(DEVICE != "cpu"),
            )
            logger.info("Loaded BGE-M3 via FlagEmbedding (multilingual, 1024-dim)")
            return _model
        except ImportError:
            logger.info("FlagEmbedding not installed, falling back to sentence-transformers")
            _USE_FLAG_EMBEDDING = False

    # Fallback: sentence-transformers (works for both bge-m3 and bge-large-en-v1.5)
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    logger.info(f"Loaded {MODEL_NAME} via sentence-transformers")
    return _model


def embed_texts(texts: list, return_sparse: bool = False) -> dict:
    """
    Generate dense embeddings for a list of texts.

    Returns:
        dict with key 'dense' — numpy array of shape (N, 1024)
    """
    if not texts:
        return {"dense": np.array([]), "sparse": []}

    model = _get_model()
    logger.debug(f"Embedding {len(texts)} texts (batch_size={BATCH_SIZE})")

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
        # Only add prefix for bge-large-en-v1.5 (bge-m3 via ST doesn't need it)
        if "bge-large" in MODEL_NAME and "m3" not in MODEL_NAME:
            texts_to_encode = [
                "Represent this sentence for searching relevant passages: " + t
                for t in texts
            ]
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
    Embed a single query string.

    Returns:
        dict with 'dense' (1D numpy array of 1024 floats) and 'sparse'.
    """
    result = embed_texts([query], return_sparse=False)
    return {
        "dense": result["dense"][0],
        "sparse": {},
    }
