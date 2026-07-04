"""Orchestrates one experiment: load → embed → build → (reps × efSearch sweep) → measure.

Implements PLAN §4 data flow exactly, including the build-once-per-rep contract and
insertion-order-shuffle repetitions.
"""

from __future__ import annotations

import time as _time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np

from .config import ExperimentConfig
from .dataset import (
    Dataset,
    download_dataset,
    generate_synthetic,
    load_jsonl,
)
from .embedding import embed_texts
from .hashing import data_hashes_for, experiment_hash, short
from .index import build_flat, build_hnsw, init_faiss
from .machine_info import collect_machine_info
from .metrics import LatencyStats, aggregate_reps, latency_stats, recall_at_k


@dataclass
class FlatResult:
    """Ground-truth Flat measurements (deterministic — no error bars)."""

    recall: float
    latency: LatencyStats
    build_time_s: float
    size_mb: float


@dataclass
class SweepPoint:
    """Aggregated HNSW measurements at one efSearch value."""

    ef_search: int
    recall_mean: float
    recall_std: float
    recall_per_rep: list[float]
    latency: LatencyStats
    build_time_s_mean: float
    size_mb: float


@dataclass
class ExperimentResult:
    """Everything the report and results.json need, including raw per-rep numbers."""

    schema_version: int
    name: str
    experiment_type: str
    short_hash: str
    full_hash: str
    timestamp: str
    duration_s: float
    k: int
    repetitions: int
    build_count: int
    synthetic: bool
    dataset_summary: dict
    machine_info: dict
    resolved_config: dict
    flat: FlatResult
    sweep: list[SweepPoint] = field(default_factory=list)

    def to_json_dict(self) -> dict:
        """JSON-serializable dict for results.json (latency dataclasses flattened)."""
        d = asdict(self)
        return d


def _resolve_dataset(
    config: ExperimentConfig, log
) -> tuple[np.ndarray, np.ndarray, str, str, dict]:
    """Return (corpus_vectors, query_vectors, corpus_hash, queries_hash, summary)."""
    ds = config.dataset
    if ds.kind == "synthetic":
        s = ds.synthetic
        assert s is not None
        log(f"  generating synthetic corpus: {s.n_docs} docs / {s.n_queries} queries / dim {s.dim}")
        syn = generate_synthetic(s.n_docs, s.n_queries, s.dim, config.seed)
        corpus_hash, queries_hash = data_hashes_for(config, None, None)
        summary = {"name": config.name, "type": "synthetic",
                   "n_docs": s.n_docs, "n_queries": s.n_queries}
        return syn.corpus_vectors, syn.query_vectors, corpus_hash, queries_hash, summary

    if ds.kind == "remote":
        r = ds.remote
        assert r is not None
        log(f"  resolving remote dataset '{r.name}' (checksum-verified, cached)")
        corpus_path, queries_path = download_dataset(
            r.name, r.url, r.sha256, r.queries_url, r.queries_sha256
        )
        summary_name = r.name
    else:
        corpus_path, queries_path = ds.corpus, ds.queries
        assert corpus_path is not None and queries_path is not None
        summary_name = config.name

    corpus = load_jsonl(corpus_path)
    queries = load_jsonl(queries_path)
    corpus_hash, queries_hash = data_hashes_for(config, corpus_path, queries_path)

    corpus_vectors = embed_texts(corpus.texts, config.embedding.model, corpus_hash, log=log)
    query_vectors = embed_texts(queries.texts, config.embedding.model, queries_hash, log=log)
    summary = {"name": summary_name, "type": ds.kind,
               "n_docs": len(corpus.ids), "n_queries": len(queries.ids)}
    return corpus_vectors, query_vectors, corpus_hash, queries_hash, summary


def _timed_search(index, query_vectors: np.ndarray, k: int, warmup: int):
    """Discard `warmup` queries, then return (ids, latencies_ns) over all queries."""
    if warmup > 0:
        w = min(warmup, query_vectors.shape[0])
        index.search(query_vectors[:w], k)
    return index.search(query_vectors, k)


def run_experiment(config: ExperimentConfig, log=print) -> ExperimentResult:
    """Run the full flat-vs-hnsw experiment and return an ExperimentResult."""
    start = datetime.now(timezone.utc)

    # Note: FAISS is forced single-threaded (init_faiss) only just before the index
    # builds, NOT here — otherwise faiss's omp_set_num_threads(1) also throttles torch and
    # embedding runs on one core. Embedding is not part of the reproducibility contract
    # (single-threaded builds are), so it is free to use all cores.
    corpus_vectors, query_vectors, corpus_hash, queries_hash, dataset_summary = (
        _resolve_dataset(config, log)
    )
    n_queries = query_vectors.shape[0]
    if n_queries < config.evaluation.min_queries:
        raise ValueError(
            f"query set has {n_queries} queries but evaluation.min_queries="
            f"{config.evaluation.min_queries}; add more queries or lower min_queries"
        )

    full_hash = experiment_hash(config, corpus_hash, queries_hash)
    short_hash = short(full_hash)
    k = config.evaluation.k
    warmup = config.evaluation.warmup_queries
    log(f"  experiment hash: {short_hash}")

    # --- Flat: ground truth ---
    # From here on, force single-threaded FAISS for reproducible, deterministic builds.
    init_faiss()
    log("  building Flat (exact) index ...")
    t0 = _time.perf_counter_ns()
    flat = build_flat(corpus_vectors)
    flat_build_s = (_time.perf_counter_ns() - t0) / 1e9
    gt_ids, flat_lat = _timed_search(flat, query_vectors, k, warmup)
    flat_result = FlatResult(
        recall=recall_at_k(gt_ids, gt_ids, k),  # self-recall == 1.0
        latency=latency_stats(flat_lat),
        build_time_s=flat_build_s,
        size_mb=flat.size_mb(),
    )
    log(f"  Flat p50={flat_result.latency.p50_ms:.3f}ms  self-recall={flat_result.recall:.3f}")

    # --- HNSW: reps × efSearch sweep ---
    n_docs = corpus_vectors.shape[0]
    ef_values = config.index.hnsw.ef_search
    # per-ef accumulators
    recalls: dict[int, list[float]] = {ef: [] for ef in ef_values}
    latencies: dict[int, list[np.ndarray]] = {ef: [] for ef in ef_values}
    build_times: list[float] = []
    hnsw_size_mb = 0.0
    build_count = 0

    for rep in range(config.evaluation.repetitions):
        rng = np.random.default_rng(config.seed + rep)
        order = rng.permutation(n_docs)
        log(f"  rep {rep}: building HNSW (insertion order seed {config.seed + rep}) ...")
        hnsw = build_hnsw(
            corpus_vectors,
            config.index.hnsw.M,
            config.index.hnsw.ef_construction,
            order,
        )
        build_count += 1
        build_times.append(hnsw.build_time_s)
        hnsw_size_mb = hnsw.size_mb()
        for ef in ef_values:
            hnsw.set_ef_search(ef)
            ids, lat = _timed_search(hnsw, query_vectors, k, warmup)
            r = recall_at_k(ids, gt_ids, k)
            recalls[ef].append(r)
            latencies[ef].append(lat)
            log(f"    ef={ef:<4d} recall@{k}={r:.4f} p50={latency_stats(lat).p50_ms:.3f}ms")

    sweep: list[SweepPoint] = []
    for ef in ef_values:
        mean, std = aggregate_reps(recalls[ef])
        pooled = np.concatenate(latencies[ef])
        sweep.append(
            SweepPoint(
                ef_search=ef,
                recall_mean=mean,
                recall_std=std,
                recall_per_rep=recalls[ef],
                latency=latency_stats(pooled),
                build_time_s_mean=float(np.mean(build_times)),
                size_mb=hnsw_size_mb,
            )
        )

    end = datetime.now(timezone.utc)
    return ExperimentResult(
        schema_version=1,
        name=config.name,
        experiment_type="flat-vs-hnsw",
        short_hash=short_hash,
        full_hash=full_hash,
        timestamp=start.isoformat(),
        duration_s=(end - start).total_seconds(),
        k=k,
        repetitions=config.evaluation.repetitions,
        build_count=build_count,
        synthetic=(config.dataset.kind == "synthetic"),
        dataset_summary=dataset_summary,
        machine_info=collect_machine_info(),
        resolved_config=config.model_dump(mode="json"),
        flat=flat_result,
        sweep=sweep,
    )
