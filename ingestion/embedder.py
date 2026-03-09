"""
Glass Expert AI — Embedding Module
Uses BAAI/bge-large-en-v1.5 for dense embeddings.
"""

import os
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
DEVICE     = os.getenv("EMBEDDING_DEVICE", "cpu")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# ── Lazy model loading (loads once on first use) ───────────────────────────────
_model = None


def _get_model():
    """Load and cache the embedding model (singleton pattern)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        logger.info(f"Device: {DEVICE.upper()}")
        logger.info("First run will download the model from HuggingFace...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        logger.info("Embedding model loaded successfully.")
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

    prefixed = [
        "Represent this sentence for searching relevant passages: " + t
        for t in texts
    ]

    dense_vecs = model.encode(
        prefixed,
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