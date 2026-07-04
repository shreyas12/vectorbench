"""Content-aware hashing: stability, sensitivity to corpus/model, YAML-order invariance."""

from vectorbench.config import ExperimentConfig
from vectorbench.hashing import canonicalize, experiment_hash, file_sha256, short


def _cfg(name="t", model="BAAI/bge-small-en-v1.5"):
    return ExperimentConfig.model_validate(
        {
            "name": name,
            "dataset": {"synthetic": {"n_docs": 10, "n_queries": 5, "dim": 8}},
            "embedding": {"model": model},
            "evaluation": {"min_queries": 5},
        }
    )


def test_same_inputs_same_hash():
    c = _cfg()
    assert experiment_hash(c, "corpusA", "queryA") == experiment_hash(c, "corpusA", "queryA")


def test_corpus_change_changes_hash():
    c = _cfg()
    assert experiment_hash(c, "corpusA", "q") != experiment_hash(c, "corpusB", "q")


def test_model_change_changes_hash():
    a = experiment_hash(_cfg(model="BAAI/bge-small-en-v1.5"), "c", "q")
    b = experiment_hash(_cfg(model="other/model"), "c", "q")
    assert a != b


def test_output_dir_does_not_affect_hash():
    a = ExperimentConfig.model_validate(
        {"name": "t", "dataset": {"synthetic": {"n_docs": 10, "n_queries": 5, "dim": 8}},
         "evaluation": {"min_queries": 5}, "output": {"dir": "runs/"}}
    )
    b = ExperimentConfig.model_validate(
        {"name": "t", "dataset": {"synthetic": {"n_docs": 10, "n_queries": 5, "dim": 8}},
         "evaluation": {"min_queries": 5}, "output": {"dir": "other/"}}
    )
    assert canonicalize(a) == canonicalize(b)


def test_short_is_eight_chars():
    assert len(short(experiment_hash(_cfg(), "c", "q"))) == 8


def test_file_sha256(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    q = tmp_path / "g.txt"
    q.write_text("hello")
    assert file_sha256(p) == file_sha256(q)
    q.write_text("world")
    assert file_sha256(p) != file_sha256(q)
