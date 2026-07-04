"""Pure metric functions: recall@k, latency percentiles, repetition aggregation.

numpy in, numbers out. No I/O, no FAISS, no pipeline knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    """Latency summary in milliseconds."""

    p50_ms: float
    p95_ms: float
    iqr_ms: float
    n: int


def recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Mean over queries of |retrieved[:k] ∩ truth[:k]| / k.

    Rows shorter than k are padded with -1 upstream; -1 is treated as "no result" and
    never counts toward an intersection. Duplicate ids within a retrieved row count once.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    retrieved = np.asarray(retrieved)
    ground_truth = np.asarray(ground_truth)
    if retrieved.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("retrieved and ground_truth must be 2D (n_queries, *)")
    if retrieved.shape[0] != ground_truth.shape[0]:
        raise ValueError("retrieved and ground_truth must have the same number of queries")
    n_queries = retrieved.shape[0]
    if n_queries == 0:
        return 0.0

    hits = 0.0
    for i in range(n_queries):
        ret = {int(x) for x in retrieved[i, :k] if int(x) != -1}
        truth = {int(x) for x in ground_truth[i, :k] if int(x) != -1}
        if not truth:
            continue
        hits += len(ret & truth) / k
    return hits / n_queries


def latency_stats(latencies_ns: np.ndarray) -> LatencyStats:
    """Compute p50/p95/IQR (ms) from an array of per-query latencies in nanoseconds."""
    lat = np.asarray(latencies_ns, dtype=np.float64)
    if lat.size == 0:
        return LatencyStats(p50_ms=0.0, p95_ms=0.0, iqr_ms=0.0, n=0)
    ms = lat / 1e6
    p25 = float(np.percentile(ms, 25))
    p50 = float(np.percentile(ms, 50))
    p75 = float(np.percentile(ms, 75))
    p95 = float(np.percentile(ms, 95))
    return LatencyStats(p50_ms=p50, p95_ms=p95, iqr_ms=p75 - p25, n=int(lat.size))


def aggregate_reps(recalls: list[float]) -> tuple[float, float]:
    """Mean and population std across repetition recalls. n==1 → std 0.0 (never NaN)."""
    if not recalls:
        raise ValueError("recalls must be non-empty")
    arr = np.asarray(recalls, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0)) if arr.size > 1 else 0.0
    return mean, std
