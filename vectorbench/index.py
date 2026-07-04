"""FAISS Flat + HNSW indexes with a build-once contract and single-threaded reproducibility.

Search returns *canonical doc ids* — the positional index into the original corpus —
so Flat (ground truth) and HNSW results are directly comparable regardless of the
insertion order HNSW was built with. The efSearch sweep mutates the live index; there
is deliberately no rebuild-per-efSearch path in this API.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field

import faiss
import numpy as np


def init_faiss() -> None:
    """Force single-threaded FAISS for reproducible, deterministic builds and timing."""
    faiss.omp_set_num_threads(1)


def _search_per_query(
    index: faiss.Index, id_map: np.ndarray, query_vectors: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run one query at a time; return (canonical_ids (n,k), latencies_ns (n,))."""
    n = query_vectors.shape[0]
    ids = np.full((n, k), -1, dtype=np.int64)
    latencies = np.empty(n, dtype=np.int64)
    for i in range(n):
        q = query_vectors[i : i + 1]
        t0 = time.perf_counter_ns()
        _, idx = index.search(q, k)
        latencies[i] = time.perf_counter_ns() - t0
        row = idx[0]
        # FAISS uses -1 for "no result"; keep those as -1, map the rest to canonical ids.
        valid = row != -1
        mapped = np.full(k, -1, dtype=np.int64)
        mapped[valid] = id_map[row[valid]]
        ids[i] = mapped
    return ids, latencies


def _index_size_mb(index: faiss.Index) -> float:
    fd, tmp = tempfile.mkstemp(suffix=".faiss")
    os.close(fd)
    try:
        faiss.write_index(index, tmp)
        size = os.path.getsize(tmp)
    finally:
        os.unlink(tmp)
    return size / (1024 * 1024)


@dataclass
class FlatIndex:
    """Exact-search ground-truth index (IndexFlatL2)."""

    _index: faiss.Index
    _id_map: np.ndarray

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return _search_per_query(self._index, self._id_map, query_vectors, k)

    def size_mb(self) -> float:
        return _index_size_mb(self._index)


@dataclass
class HNSWIndex:
    """Approximate HNSW index. efSearch is mutated in place; never rebuilt for the sweep."""

    _index: faiss.Index
    _id_map: np.ndarray
    build_time_s: float
    insertion_order: np.ndarray = field(repr=False)

    def set_ef_search(self, ef_search: int) -> None:
        """The ONLY way the sweep changes the index — mutates efSearch on the live graph."""
        self._index.hnsw.efSearch = int(ef_search)

    def search(self, query_vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return _search_per_query(self._index, self._id_map, query_vectors, k)

    def size_mb(self) -> float:
        return _index_size_mb(self._index)


def build_flat(vectors: np.ndarray) -> FlatIndex:
    """Build an exact IndexFlatL2 over vectors in original order (canonical id == position)."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    id_map = np.arange(vectors.shape[0], dtype=np.int64)
    return FlatIndex(_index=index, _id_map=id_map)


def build_hnsw(
    vectors: np.ndarray,
    M: int,
    ef_construction: int,
    insertion_order: np.ndarray,
) -> HNSWIndex:
    """Build an IndexHNSWFlat once, adding vectors in `insertion_order`.

    The order→canonical-id mapping is preserved so search returns original corpus
    positions. Single-threaded (see `init_faiss`) makes the build deterministic given
    the insertion order.
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    insertion_order = np.asarray(insertion_order, dtype=np.int64)
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = int(ef_construction)

    reordered = np.ascontiguousarray(vectors[insertion_order])
    t0 = time.perf_counter_ns()
    index.add(reordered)
    build_time_s = (time.perf_counter_ns() - t0) / 1e9

    # FAISS position p corresponds to the vector added at p, i.e. original index order[p].
    id_map = insertion_order.copy()
    return HNSWIndex(
        _index=index,
        _id_map=id_map,
        build_time_s=build_time_s,
        insertion_order=insertion_order,
    )
