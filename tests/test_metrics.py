"""Metric correctness: recall@k edge cases, latency stats, repetition aggregation."""

import numpy as np
import pytest

from vectorbench.metrics import aggregate_reps, latency_stats, recall_at_k


def test_perfect_recall():
    ret = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(ret, ret, 3) == 1.0


def test_zero_recall():
    ret = np.array([[1, 2, 3]])
    truth = np.array([[7, 8, 9]])
    assert recall_at_k(ret, truth, 3) == 0.0


def test_partial_recall():
    ret = np.array([[1, 2, 9]])
    truth = np.array([[1, 2, 3]])
    assert recall_at_k(ret, truth, 3) == pytest.approx(2 / 3)


def test_duplicate_ids_counted_once():
    ret = np.array([[1, 1, 1]])
    truth = np.array([[1, 2, 3]])
    # only one distinct hit out of k=3
    assert recall_at_k(ret, truth, 3) == pytest.approx(1 / 3)


def test_k_larger_than_result_set_pads_with_neg1():
    ret = np.array([[1, 2, -1, -1]])
    truth = np.array([[1, 2, 3, 4]])
    # 2 hits, k=4
    assert recall_at_k(ret, truth, 4) == pytest.approx(2 / 4)


def test_neg1_never_counts_as_hit():
    ret = np.array([[-1, -1]])
    truth = np.array([[-1, 5]])
    assert recall_at_k(ret, truth, 2) == 0.0


def test_empty_query_set():
    ret = np.empty((0, 3), dtype=int)
    truth = np.empty((0, 3), dtype=int)
    assert recall_at_k(ret, truth, 3) == 0.0


def test_ties_do_not_double_count():
    ret = np.array([[1, 2, 3, 3]])
    truth = np.array([[1, 2, 3, 4]])
    assert recall_at_k(ret, truth, 4) == pytest.approx(3 / 4)


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        recall_at_k(np.array([[1]]), np.array([[1]]), 0)


def test_latency_stats_basic():
    # 1..100 ms in ns
    ns = (np.arange(1, 101) * 1e6).astype(np.int64)
    s = latency_stats(ns)
    assert s.n == 100
    assert s.p50_ms == pytest.approx(50.5, abs=0.6)
    assert s.p95_ms == pytest.approx(95.05, abs=1.0)
    assert s.iqr_ms > 0


def test_latency_stats_empty():
    s = latency_stats(np.array([], dtype=np.int64))
    assert s.n == 0 and s.p50_ms == 0.0


def test_aggregate_single_rep_std_zero():
    mean, std = aggregate_reps([0.7])
    assert mean == 0.7 and std == 0.0


def test_aggregate_multi_rep():
    mean, std = aggregate_reps([0.6, 0.8])
    assert mean == pytest.approx(0.7)
    assert std == pytest.approx(0.1)


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_reps([])
