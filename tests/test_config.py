"""Config validation: valid parse, unknown/missing fields, dataset-form and value rules."""

import textwrap

import pytest

from vectorbench.config import ConfigError, load_config


def _write(tmp_path, text, name="c.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


def test_valid_synthetic_config(tmp_path):
    p = _write(tmp_path, """
        name: t
        dataset:
          synthetic:
            n_docs: 200
            n_queries: 20
            dim: 32
        evaluation:
          min_queries: 20
    """)
    cfg = load_config(p)
    assert cfg.name == "t"
    assert cfg.dataset.kind == "synthetic"
    assert cfg.similarity == "cosine"
    assert cfg.index.hnsw.ef_search  # defaults present


def test_unknown_field_rejected(tmp_path):
    p = _write(tmp_path, """
        name: t
        dataset:
          synthetic: {n_docs: 10, n_queries: 5, dim: 8}
        bogus: 1
    """)
    with pytest.raises(ConfigError):
        load_config(p)


def test_similarity_dot_rejected(tmp_path):
    p = _write(tmp_path, """
        name: t
        similarity: dot
        dataset:
          synthetic: {n_docs: 10, n_queries: 5, dim: 8}
    """)
    with pytest.raises(ConfigError) as e:
        load_config(p)
    assert "cosine" in str(e.value)


def test_repetitions_zero_rejected(tmp_path):
    p = _write(tmp_path, """
        name: t
        dataset:
          synthetic: {n_docs: 10, n_queries: 5, dim: 8}
        evaluation: {repetitions: 0}
    """)
    with pytest.raises(ConfigError):
        load_config(p)


def test_two_dataset_forms_rejected(tmp_path):
    p = _write(tmp_path, """
        name: t
        dataset:
          synthetic: {n_docs: 10, n_queries: 5, dim: 8}
          corpus: /nope/corpus.jsonl
          queries: /nope/queries.jsonl
    """)
    with pytest.raises(ConfigError):
        load_config(p)


def test_no_dataset_form_rejected(tmp_path):
    p = _write(tmp_path, """
        name: t
        dataset: {}
    """)
    with pytest.raises(ConfigError):
        load_config(p)


def test_missing_local_file_message_names_path(tmp_path):
    p = _write(tmp_path, """
        name: t
        dataset:
          corpus: /definitely/missing/corpus.jsonl
          queries: /definitely/missing/queries.jsonl
    """)
    with pytest.raises(ConfigError) as e:
        load_config(p)
    assert "corpus" in str(e.value) and "missing" in str(e.value)


def test_empty_ef_search_rejected(tmp_path):
    p = _write(tmp_path, """
        name: t
        dataset:
          synthetic: {n_docs: 10, n_queries: 5, dim: 8}
        index:
          hnsw: {ef_search: []}
    """)
    with pytest.raises(ConfigError):
        load_config(p)


def test_missing_config_file():
    with pytest.raises(ConfigError):
        load_config("/no/such/config.yaml")
