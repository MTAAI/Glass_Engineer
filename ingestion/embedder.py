"""
Glass Expert AI — Embedding Module
===================================
Model:   BAAI/bge-m3  (multilingual, 1024-dim dense + sparse lexical weights)
GPU:     RTX PRO 5000 Blackwell — FP16, standard attention
         (sm_120 not yet in flash-attn; eager mode is correct here)

Key improvement over both branches:
  - Sparse lexical weights from bge-m3 ARE returned and used for true
    hybrid search via RRF fusion. Neither branch currently does this:
    Engineer Z discards sparse and uses ILIKE; Arjun branch discards
    sparse entirely.
  - bge-m3 sparse weights are far more accurate than ILIKE for chemical
    formulas (SiO2, Na2O, B2O3) and glass terminology.
  - GPU-first defaults: EMBEDDING_DEVICE=cuda, USE_FP16=true.

Two load paths:
  1. FlagEmbedding (preferred) — exposes sparse weights + faster GPU
  2. sentence-transformers (fallback) — dense only, no sparse

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

# Limit concurrent GPU operations — prevents OOM and timeouts under load
_GPU_SEMAPHORE = threading.Semaphore(4)

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("EMBEDDING_MODEL",      "BAAI/bge-m3")
DEVICE     = os.getenv("EMBEDDING_DEVICE",     "cuda")
USE_FP16   = os.getenv("EMBEDDING_USE_FP16",   "true").lower() == "true"
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))

# Instruction prefix improves retrieval quality per bge-m3 model card
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Module-level flag — set after first load
_use_flag_embedding = True


@lru_cache(maxsize=1)
def _load_model():
    """Load bge-m3 once. lru_cache(1) = true singleton."""
    global _use_flag_embedding

    logger.info(
        f"Loading embedding model: {MODEL_NAME} | "
        f"device={DEVICE} | fp16={USE_FP16}"
    )

    # ── Preferred path: FlagEmbedding (exposes sparse weights) ────────────────
    try:
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(
            MODEL_NAME,
            use_fp16=USE_FP16,
            device=DEVICE,
        )
        _use_flag_embedding = True
        logger.info("bge-m3 loaded via FlagEmbedding — dense + sparse available")
        return model
    except ImportError:
        logger.warning(
            "FlagEmbedding not installed — sparse weights unavailable. "
            "Install with: pip install FlagEmbedding"
        )
    except Exception as e:
        logger.warning(f"FlagEmbedding failed ({e}) — falling back to sentence-transformers")

    # ── Fallback: sentence-transformers (dense only) ───────────────────────────
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    _use_flag_embedding = False
    logger.info("bge-m3 loaded via sentence-transformers — dense only")
    return model


def _get_model():
    return _load_model()


def _normalise(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise a batch of vectors."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


# ── Batch embedding — ingestion path ──────────────────────────────────────────
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
    bs    = batch_size or BATCH_SIZE

    logger.debug(
        f"Embedding {len(texts)} texts | "
        f"batch={bs} | sparse={return_sparse and _use_flag_embedding}"
    )

    with _GPU_SEMAPHORE:
        if _use_flag_embedding:
            result = model.encode(
                texts,
                return_dense=True,
                return_sparse=return_sparse,
                return_colbert_vecs=False,
                batch_size=bs,
                max_length=MAX_LENGTH,
                show_progress_bar=show_progress,
            )
            dense  = _normalise(result["dense_vecs"])
            sparse = result.get("lexical_weights", []) if return_sparse else []
            return {"dense": dense, "sparse": sparse}

        else:
            dense = model.encode(
                texts,
                batch_size=bs,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            )
            return {"dense": dense, "sparse": []}


# ── Single-query embedding — inference path ────────────────────────────────────
def embed_query(query: str) -> dict:
    """
    Embed a single user query.
    GPU semaphore limits concurrent inference to 4 to prevent OOM under load.

    Returns:
        {
          "dense":  np.ndarray (1024,)
          "sparse": dict[str, float]  — empty dict if FlagEmbedding unavailable
        }
    """
    prefixed = _QUERY_PREFIX + query

    with _GPU_SEMAPHORE:
        if _use_flag_embedding:
            model  = _get_model()
            result = model.encode(
                [prefixed],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
                batch_size=1,
                max_length=MAX_LENGTH,
            )
            dense  = _normalise(result["dense_vecs"])[0]
            sparse = result["lexical_weights"][0] if result.get("lexical_weights") else {}
            return {"dense": dense, "sparse": sparse}

        else:
            result = embed_texts([prefixed], return_sparse=False, batch_size=1)
            return {"dense": result["dense"][0], "sparse": {}}


# ── Utilities ──────────────────────────────────────────────────────────────────
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