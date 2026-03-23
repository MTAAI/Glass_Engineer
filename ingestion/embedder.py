"""
Glass Expert AI — Embedding Module (Master)
=============================================
Model:   BAAI/bge-m3  (multilingual, 1024-dim dense + sparse lexical weights)
GPU:     RTX PRO 5000 — FP16, standard attention

Features merged from both branches:
  - FlagEmbedding preferred (exposes sparse weights for hybrid search)
  - sentence-transformers fallback (dense only)
  - GPU semaphore — limits concurrent inference to 4 (prevents OOM under load)
  - Warmup function for startup pre-loading
  - Supports both bge-m3 (multilingual) and bge-large-en-v1.5 (legacy)

Usage:
    from ingestion.embedder import embed_query, embed_texts, warmup
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Optional

import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEVICE     = os.getenv("EMBEDDING_DEVICE", "cpu")
USE_FP16   = os.getenv("EMBEDDING_USE_FP16", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))

# GPU semaphore — limits concurrent inference to prevent OOM under load
# 4 is optimal for RTX PRO 5000 (16GB VRAM) with bge-m3 (1.7GB model)
_GPU_SEMAPHORE = threading.Semaphore(int(os.getenv("EMBEDDING_MAX_CONCURRENT", "4")))

# BGE-M3 can use FlagEmbedding for best performance (exposes sparse weights)
_USE_FLAG_EMBEDDING = MODEL_NAME == "BAAI/bge-m3"

# Query prefix per bge-m3 model card — improves retrieval quality
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ── Lazy model loading (loads once on first use) ──────────────────────────────
@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the embedding model (singleton via lru_cache)."""
    global _USE_FLAG_EMBEDDING

    logger.info(f"Loading embedding model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE.upper()} | FP16: {USE_FP16}")

    if _USE_FLAG_EMBEDDING:
        try:
            from FlagEmbedding import BGEM3FlagModel
            model = BGEM3FlagModel(
                MODEL_NAME,
                use_fp16=USE_FP16,
                device=DEVICE,
            )
            logger.info("bge-m3 loaded via FlagEmbedding — dense + sparse available")
            return model
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed — sparse weights unavailable. "
                "Install with: pip install FlagEmbedding"
            )
            _USE_FLAG_EMBEDDING = False
        except Exception as e:
            logger.warning(f"FlagEmbedding failed ({e}) — falling back to sentence-transformers")
            _USE_FLAG_EMBEDDING = False

    # Fallback: sentence-transformers (works for both bge-m3 and bge-large-en-v1.5)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    logger.info(f"Loaded {MODEL_NAME} via sentence-transformers — dense only")
    return model


def _get_model():
    """Public accessor for the embedding model."""
    return _load_model()


def _normalise(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise a batch of vectors."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


# ── Batch embedding — ingestion path ─────────────────────────────────────────
def embed_texts(
    texts: list[str],
    return_sparse: bool = True,
    batch_size: Optional[int] = None,
    show_progress: bool = False,
) -> dict:
    """
    Embed a list of texts for ingestion.

    Returns:
        {
          "dense":  np.ndarray (N, 1024) — L2-normalised
          "sparse": list[dict[str, float]] — bge-m3 lexical weights per text
                    (empty list if FlagEmbedding unavailable)
        }
    """
    if not texts:
        return {"dense": np.array([]), "sparse": []}

    model = _get_model()
    bs = batch_size or BATCH_SIZE

    logger.debug(
        f"Embedding {len(texts)} texts | batch={bs} | "
        f"sparse={return_sparse and _USE_FLAG_EMBEDDING}"
    )

    with _GPU_SEMAPHORE:
        if _USE_FLAG_EMBEDDING:
            result = model.encode(
                texts,
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
                batch_size=bs,
                max_length=MAX_LENGTH,
                show_progress_bar=show_progress,
            )
            dense = _normalise(result["dense_vecs"])
            sparse = result.get("lexical_weights", []) if return_sparse else []
            return {"dense": dense, "sparse": sparse}

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

            dense = model.encode(
                texts_to_encode,
                batch_size=bs,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            return {"dense": dense, "sparse": []}


# ── Single-query embedding — inference path ───────────────────────────────────
def embed_query(query: str) -> dict:
    """
    Embed a single user query.
    GPU semaphore limits concurrent inference to prevent OOM under load.

    Returns:
        {
          "dense":  np.ndarray (1024,)
          "sparse": dict[str, float] — empty dict if FlagEmbedding unavailable
        }
    """
    with _GPU_SEMAPHORE:
        if _USE_FLAG_EMBEDDING:
            model = _get_model()
            prefixed = _QUERY_PREFIX + query
            result = model.encode(
                [prefixed],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
                batch_size=1,
                max_length=MAX_LENGTH,
            )
            dense = _normalise(result["dense_vecs"])[0]
            sparse = result["lexical_weights"][0] if result.get("lexical_weights") else {}
            return {"dense": dense, "sparse": sparse}

        else:
            result = embed_texts([query], return_sparse=False, batch_size=1)
            return {"dense": result["dense"][0], "sparse": {}}


# ── Utilities ─────────────────────────────────────────────────────────────────
def get_embedding_dim() -> int:
    """bge-m3 always produces 1024-dimensional vectors."""
    return 1024


def warmup() -> None:
    """
    Pre-load the model at startup so the first request is not slow.
    Call from the FastAPI lifespan handler.
    """
    logger.info("Embedding model warmup starting...")
    embed_query("glass transition temperature borosilicate warmup")
    logger.info("Embedding model warmup complete")
