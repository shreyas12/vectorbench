"""Dataset resolution: JSONL loader, synthetic vector generator, checksum-verified downloader."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm


def cache_root() -> Path:
    """Base cache directory (~/.cache/vectorbench), overridable via VECTORBENCH_CACHE."""
    env = os.environ.get("VECTORBENCH_CACHE")
    root = Path(env) if env else Path.home() / ".cache" / "vectorbench"
    return root


@dataclass(frozen=True)
class Dataset:
    """A text corpus or query set: parallel lists of ids and texts."""

    ids: list[str]
    texts: list[str]


@dataclass(frozen=True)
class SyntheticDataset:
    """Pre-embedded synthetic vectors (bypasses the embedding stage)."""

    corpus_ids: list[str]
    corpus_vectors: np.ndarray  # (n_docs, dim) float32, L2-normalized
    query_ids: list[str]
    query_vectors: np.ndarray  # (n_queries, dim) float32, L2-normalized


def load_jsonl(path: Path) -> Dataset:
    """Load a `{id, text}`-per-line JSONL file, raising with a line number on malformed rows."""
    path = Path(path)
    ids: list[str] = []
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} line {lineno}: invalid JSON: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path.name} line {lineno}: expected a JSON object")
            if "id" not in obj:
                raise ValueError(f"{path.name} line {lineno}: missing 'id' field")
            if "text" not in obj:
                raise ValueError(f"{path.name} line {lineno}: missing 'text' field")
            ids.append(str(obj["id"]))
            texts.append(str(obj["text"]))
    if not ids:
        raise ValueError(f"{path.name}: no records found")
    return Dataset(ids=ids, texts=texts)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def generate_synthetic(
    n_docs: int, n_queries: int, dim: int, seed: int
) -> SyntheticDataset:
    """Generate deterministic random-normal, L2-normalized vectors; cache as .npy.

    Same (n_docs, n_queries, dim, seed) → identical arrays. Second call reads the cache.
    """
    cache_dir = cache_root() / f"synthetic_{n_docs}_{n_queries}_{dim}_{seed}"
    corpus_npy = cache_dir / "corpus.npy"
    queries_npy = cache_dir / "queries.npy"

    if corpus_npy.exists() and queries_npy.exists():
        corpus_vectors = np.load(corpus_npy)
        query_vectors = np.load(queries_npy)
    else:
        rng = np.random.default_rng(seed)
        corpus_vectors = _normalize(rng.standard_normal((n_docs, dim)).astype(np.float32))
        query_vectors = _normalize(rng.standard_normal((n_queries, dim)).astype(np.float32))
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(corpus_npy, corpus_vectors)
        np.save(queries_npy, query_vectors)

    corpus_ids = [f"doc_{i}" for i in range(n_docs)]
    query_ids = [f"query_{i}" for i in range(n_queries)]
    return SyntheticDataset(
        corpus_ids=corpus_ids,
        corpus_vectors=corpus_vectors,
        query_ids=query_ids,
        query_vectors=query_vectors,
    )


def _sha256_stream(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(url: str, sha256: str, dest: Path, label: str) -> Path:
    """Download a single file to `dest`, verifying SHA256; gunzip if the URL is .gz.

    Uses a temp file + atomic rename so an interrupted download never poisons the cache.
    """
    if dest.exists() and _sha256_stream(dest) == sha256:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    is_gz = url.endswith(".gz")
    raw_tmp = tmp.with_suffix(tmp.suffix + ".gz") if is_gz else tmp

    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(raw_tmp, "wb") as out, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"downloading {label}"
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    out.write(chunk)
                    bar.update(len(chunk))
    except requests.RequestException as exc:
        for p in (raw_tmp, tmp):
            p.unlink(missing_ok=True)
        raise RuntimeError(
            f"this config requires a one-time download for '{label}'; "
            "check your connection or use examples/scifact-small.yaml for a fully offline run "
            f"(underlying error: {exc})"
        ) from exc

    if is_gz:
        with gzip.open(raw_tmp, "rb") as gz, open(tmp, "wb") as out:
            shutil.copyfileobj(gz, out)
        raw_tmp.unlink(missing_ok=True)

    got = _sha256_stream(tmp)
    if got != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"checksum mismatch for {label}: expected {sha256}, got {got} — "
            "the file may be corrupted or the URL changed"
        )
    os.replace(tmp, dest)
    return dest


def download_dataset(
    name: str,
    url: str,
    sha256: str,
    queries_url: str,
    queries_sha256: str,
) -> tuple[Path, Path]:
    """Fetch (corpus, queries) for a remote dataset into ~/.cache/vectorbench/datasets/<name>/."""
    dest_dir = cache_root() / "datasets" / name
    corpus_path = dest_dir / "corpus.jsonl"
    queries_path = dest_dir / "queries.jsonl"
    _download_one(url, sha256, corpus_path, f"{name} corpus")
    _download_one(queries_url, queries_sha256, queries_path, f"{name} queries")
    return corpus_path, queries_path
