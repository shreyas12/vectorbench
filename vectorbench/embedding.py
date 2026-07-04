"""BGE-small text embedding with L2 normalization and an on-disk cache.

sentence-transformers/torch are imported lazily so synthetic-only runs (and the test
suite) never need them.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .dataset import cache_root


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name)


def _assert_normalized(vectors: np.ndarray) -> None:
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        worst = float(np.max(np.abs(norms - 1.0)))
        raise ValueError(
            f"embeddings are not L2-normalized (max deviation {worst:.4g}); "
            "the L2==cosine invariant would be violated"
        )


def embed_texts(
    texts: list[str], model_name: str, cache_key: str, *, log=print
) -> np.ndarray:
    """Embed texts to float32, L2-normalized vectors, caching by corpus content hash.

    `cache_key` is the corpus content SHA. The cache path is keyed by model + cache_key,
    so re-runs on the same corpus and model skip embedding entirely.
    """
    cache_dir = cache_root() / "embeddings" / _model_slug(model_name)
    cache_path = cache_dir / f"{cache_key}.npy"
    if cache_path.exists():
        log(f"  embedding cache hit: {cache_path}")
        vectors = np.load(cache_path)
        _assert_normalized(vectors)
        return vectors.astype(np.float32)

    # Lazy heavy import — only when we actually need to embed.
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    log(f"  embedding {len(texts)} texts with {model_name} ...")
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 1000,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Guard: some model/config combos may not normalize; enforce the invariant.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.allclose(norms, 1.0, atol=1e-3):
        norms[norms == 0] = 1.0
        vectors = (vectors / norms).astype(np.float32)
    _assert_normalized(vectors)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return vectors


def embedding_dim(model_name: str) -> int:
    """Best-effort embedding dimension (used only for messaging); 384 for bge-small."""
    if "bge-small" in model_name:
        return 384
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return int(SentenceTransformer(model_name).get_sentence_embedding_dimension())
