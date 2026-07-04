"""Content-aware experiment hashing over the validated config, corpus, and versions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss

from .config import ExperimentConfig


def canonicalize(config: ExperimentConfig) -> str:
    """Deterministic serialization of the validated model (sorted keys, no whitespace noise)."""
    data = config.model_dump(mode="json", exclude={"output"})
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    """Streaming SHA256 of a file's bytes, memory-safe for large corpora."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _synthetic_signature(config: ExperimentConfig) -> str:
    s = config.dataset.synthetic
    assert s is not None
    return f"synthetic:{s.n_docs}:{s.n_queries}:{s.dim}:{config.seed}"


def experiment_hash(
    config: ExperimentConfig, corpus_hash: str, queries_hash: str
) -> str:
    """Full experiment hash over canonical config + data hashes + model + FAISS version.

    For synthetic datasets, the caller passes the synthetic signature for both data
    hashes (see `data_hashes_for`).
    """
    parts = [
        canonicalize(config),
        corpus_hash,
        queries_hash,
        config.embedding.model,
        faiss.__version__,
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def data_hashes_for(config: ExperimentConfig, corpus_path: Path | None,
                    queries_path: Path | None) -> tuple[str, str]:
    """Resolve (corpus_hash, queries_hash) for any dataset kind.

    Synthetic datasets have no files, so both hashes are the synthetic signature.
    """
    if config.dataset.kind == "synthetic":
        sig = _synthetic_signature(config)
        return sig, sig
    assert corpus_path is not None and queries_path is not None
    return file_sha256(corpus_path), file_sha256(queries_path)


def short(hash_hex: str) -> str:
    """First 8 chars of a hex hash, for headers and folder names."""
    return hash_hex[:8]
