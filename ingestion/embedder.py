"""
Glass Expert AI — Embedding Module
Model: BAAI/bge-m3  (multilingual, 1024-dim dense + sparse lexical weights)
GPU:   RTX PRO 5000 Blackwell — FP16, standard attention (no flash-attn on sm_120)
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Optional
import numpy as np
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL",      "BAAI/bge-m3")
DEVICE     = os.getenv("EMBEDDING_DEVICE",     "cuda")
USE_FP16   = os.getenv("EMBEDDING_USE_FP16",   "true").lower() == "true"
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
MAX_LENGTH = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_use_flag_embedding = True


@lru_cache(maxsize=1)
def _load_model():
    global _use_flag_embedding
    logger.info(f"Loading {MODEL_NAME} | device={DEVICE} | fp16={USE_FP16}")

    try:
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(MODEL_NAME, use_fp16=USE_FP16, device=DEVICE)
        _use_flag_embedding = True
        logger.info("bge-m3 loaded via FlagEmbedding — dense + sparse available")
        return model
    except ImportError:
        logger.warning("FlagEmbedding not installed — sparse unavailable. pip install FlagEmbedding")
    except Exception as e:
        logger.warning(f"FlagEmbedding failed ({e}) — falling back to sentence-transformers")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    _use_flag_embedding = False
    logger.info("bge-m3 loaded via sentence-transformers — dense only")
    return model


def _get_model():
    return _load_model()


def _normalise(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def embed_texts(
    texts: list[str],
    return_sparse: bool = True,
    batch_size: Optional[int] = None,
    show_progress: bool = False,
) -> dict:
    """Embed a list of texts. Returns dense (N,1024) and sparse weights."""
    if not texts:
        return {"dense": np.array([]), "sparse": []}

    model = _get_model()
    bs = batch_size or BATCH_SIZE

    if _use_flag_embedding:
        result = model.encode(
            texts,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
            batch_size=bs,
            max_length=MAX_LENGTH,
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


# Separate sentence-transformers model for queries
# (documents were stored with sentence-transformers, so queries must match)
_st_model = None

def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence-transformers for query embedding: {MODEL_NAME}")
        _st_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        logger.info("sentence-transformers query model loaded")
    return _st_model


def embed_query(query: str) -> dict:
    """
    Embed a single query using sentence-transformers WITH instruction prefix.
    Documents were ingested with this prefix so queries must match exactly.
    """
    prefixed = "Represent this sentence for searching relevant passages: " + query
    model = _get_st_model()
    vec = model.encode([prefixed], normalize_embeddings=True)[0]
    return {"dense": vec, "sparse": {}}

def get_embedding_dim() -> int:
    return 1024


def warmup() -> None:
    logger.info("Embedding warmup starting...")
    embed_query("glass transition temperature warmup")
    logger.info("Embedding warmup complete")