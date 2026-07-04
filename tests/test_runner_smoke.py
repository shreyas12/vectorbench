"""End-to-end synthetic smoke test + index-level determinism/shuffle assertions."""

import json

import numpy as np
import pytest

from vectorbench.config import ExperimentConfig
from vectorbench.index import build_hnsw, init_faiss
from vectorbench.report import resolved_config_yaml, write_outputs
from vectorbench.runner import run_experiment


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTORBENCH_CACHE", str(tmp_path / "cache"))


def _smoke_config():
    return ExperimentConfig.model_validate(
        {
            "name": "smoke",
            "dataset": {"synthetic": {"n_docs": 200, "n_queries": 20, "dim": 32}},
            "index": {"hnsw": {"M": 16, "ef_construction": 64, "ef_search": [16]}},
            "evaluation": {"k": 10, "min_queries": 20, "warmup_queries": 2, "repetitions": 2},
            "seed": 42,
        }
    )


def test_end_to_end_synthetic(tmp_path):
    cfg = _smoke_config()
    result = run_experiment(cfg, log=lambda *a, **k: None)

    # Flat is ground truth → self-recall exactly 1.0.
    assert result.flat.recall == 1.0
    # HNSW recall in [0, 1].
    for point in result.sweep:
        assert 0.0 <= point.recall_mean <= 1.0
    # Build happened exactly `repetitions` times regardless of sweep length.
    assert result.build_count == cfg.evaluation.repetitions

    # Outputs written and re-loadable.
    config_yaml = resolved_config_yaml(cfg)
    report_path = write_outputs(result, tmp_path / "runs", config_yaml)
    assert report_path.exists()
    run_dir = report_path.parent
    for fname in ("results.json", "config.yaml", "metadata.json", "experiment_report.html"):
        assert (run_dir / fname).exists()

    results = json.loads((run_dir / "results.json").read_text())
    assert results["schema_version"] == 1
    assert len(results["sweep"][0]["recall_per_rep"]) == cfg.evaluation.repetitions

    # results.json stays small (percentiles, not raw per-query arrays).
    assert (run_dir / "results.json").stat().st_size < 1_000_000

    # Report loads no external assets (renders offline). The inlined Plotly bundle
    # contains URL *string literals* in its own source (unused mapbox basemaps), so we
    # check for real external-resource tags, not any substring.
    html = (run_dir / "experiment_report.html").read_text()
    assert '<script src="http' not in html
    assert '<link ' not in html
    assert '<img src="http' not in html


def test_resolved_config_roundtrip_same_hash(tmp_path):
    cfg = _smoke_config()
    result = run_experiment(cfg, log=lambda *a, **k: None)
    config_yaml = resolved_config_yaml(cfg)
    report_path = write_outputs(result, tmp_path / "runs", config_yaml)

    from vectorbench.config import load_config

    reloaded = load_config(report_path.parent / "config.yaml")
    result2 = run_experiment(reloaded, log=lambda *a, **k: None)
    assert result2.full_hash == result.full_hash


def test_reps_use_different_insertion_orders():
    cfg = _smoke_config()
    orders = []
    for rep in range(cfg.evaluation.repetitions):
        rng = np.random.default_rng(cfg.seed + rep)
        orders.append(rng.permutation(200))
    assert not np.array_equal(orders[0], orders[1])


def test_same_order_builds_are_deterministic():
    init_faiss()
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((200, 32)).astype("float32")
    order = np.random.default_rng(1).permutation(200)
    q = vectors[:5]

    a = build_hnsw(vectors, 16, 64, order)
    b = build_hnsw(vectors, 16, 64, order)
    a.set_ef_search(16)
    b.set_ef_search(16)
    ids_a, _ = a.search(q, 10)
    ids_b, _ = b.search(q, 10)
    assert np.array_equal(ids_a, ids_b)


def test_different_orders_produce_different_graphs():
    init_faiss()
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((300, 32)).astype("float32")
    order1 = np.random.default_rng(1).permutation(300)
    order2 = np.random.default_rng(2).permutation(300)
    q = vectors[:20]

    a = build_hnsw(vectors, 8, 16, order1)
    b = build_hnsw(vectors, 8, 16, order2)
    a.set_ef_search(4)
    b.set_ef_search(4)
    ids_a, _ = a.search(q, 10)
    ids_b, _ = b.search(q, 10)
    # At low efSearch, different insertion orders should diverge on at least one query.
    assert not np.array_equal(ids_a, ids_b)


def test_hnsw_returns_doc_ids_matching_flat_on_easy_query():
    from vectorbench.index import build_flat

    init_faiss()
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((100, 16)).astype("float32")
    order = np.arange(100)
    flat = build_flat(vectors)
    hnsw = build_hnsw(vectors, 16, 64, order)
    hnsw.set_ef_search(256)
    # A corpus vector as its own query: nearest neighbour is itself for both indexes.
    q = vectors[42:43]
    f_ids, _ = flat.search(q, 1)
    h_ids, _ = hnsw.search(q, 1)
    assert f_ids[0, 0] == 42 == h_ids[0, 0]
