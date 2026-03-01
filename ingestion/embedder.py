"""
Glass Expert AI — Embedding Module
Uses BAAI/bge-m3 for multilingual dense + sparse embeddings.
Optimised for NVIDIA RTX PRO 5000 Blackwell (32GB VRAM).
"""

import os
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEVICE       = os.getenv("EMBEDDING_DEVICE", "cpu")
USE_FP16     = os.getenv("EMBEDDING_USE_FP16", "false").lower() == "true"
BATCH_SIZE   = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
MAX_LENGTH   = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))

# ── Lazy model loading (loads once on first use) ───────────────────────────────
_model = None


def _get_model():
    """Load and cache the bge-m3 model (singleton pattern)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        logger.info(f"Device: {DEVICE.upper()} | FP16: {USE_FP16}")
        logger.info("First run will download ~2.2 GB from HuggingFace...")

        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel(
            MODEL_NAME,
            use_fp16=USE_FP16,
            device=DEVICE,
        )
        logger.info("✅ Embedding model loaded successfully.")
    return _model


def embed_texts(
    texts: list[str],
    return_sparse: bool = True,
) -> dict:
    """
    Generate dense (and optionally sparse) embeddings for a list of texts.

    Args:
        texts:          List of text strings to embed.
        return_sparse:  Whether to also return sparse (keyword) embeddings.

    Returns:
        dict with keys:
            'dense'  — numpy array of shape (N, 1024)
            'sparse' — list of dicts {token: weight} if return_sparse=True
    """
    if not texts:
        return {"dense": np.array([]), "sparse": []}

    model = _get_model()

    logger.debug(f"Embedding {len(texts)} texts (batch_size={BATCH_SIZE})")

    result = model.encode(
        texts,
        return_dense=True,
        return_sparse=return_sparse,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    output = {
        "dense": result["dense_vecs"],  # shape: (N, 1024)
    }

    if return_sparse:
        output["sparse"] = result.get("lexical_weights", [])

    return output


def embed_query(query: str) -> dict:
    """
    Embed a single query string.
    Uses a slightly different instruction prefix for better retrieval quality.

    Args:
        query: The user's question or search query.

    Returns:
        dict with 'dense' (1D numpy array of 1024 floats) and 'sparse'.
    """
    # bge-m3 performs better with this instruction prefix for queries
    instruction = "Represent this sentence for searching relevant passages: "
    prefixed_query = instruction + query

    result = embed_texts([prefixed_query], return_sparse=True)

    return {
        "dense": result["dense"][0],          # 1D array (1024,)
        "sparse": result["sparse"][0] if result["sparse"] else {},
    }
