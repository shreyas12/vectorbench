"""Downloader: fresh download, cached hit, and checksum-failure paths — no network."""

import gzip
import hashlib

import pytest

from vectorbench.dataset import _download_one, cache_root


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTORBENCH_CACHE", str(tmp_path / "cache"))


def _serve(monkeypatch, payload: bytes):
    """Patch requests.get to stream `payload` without touching the network."""

    class _Resp:
        headers = {"content-length": str(len(payload))}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=1):
            for i in range(0, len(payload), chunk_size):
                yield payload[i : i + chunk_size]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("vectorbench.dataset.requests.get", lambda *a, **k: _Resp())


def test_fresh_download_and_checksum_ok(tmp_path, monkeypatch):
    payload = b'{"id": "1", "text": "hello"}\n'
    _serve(monkeypatch, payload)
    sha = hashlib.sha256(payload).hexdigest()
    dest = cache_root() / "datasets" / "x" / "corpus.jsonl"
    out = _download_one("http://fake/corpus.jsonl", sha, dest, "x corpus")
    assert out.read_bytes() == payload


def test_cached_hit_skips_download(tmp_path, monkeypatch):
    payload = b"cached-content\n"
    sha = hashlib.sha256(payload).hexdigest()
    dest = cache_root() / "datasets" / "x" / "corpus.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)

    def _boom(*a, **k):
        raise AssertionError("network should not be hit on a cached, valid file")

    monkeypatch.setattr("vectorbench.dataset.requests.get", _boom)
    out = _download_one("http://fake/corpus.jsonl", sha, dest, "x corpus")
    assert out.read_bytes() == payload


def test_checksum_mismatch_raises_and_leaves_no_file(tmp_path, monkeypatch):
    payload = b"actual-bytes\n"
    _serve(monkeypatch, payload)
    wrong_sha = "0" * 64
    dest = cache_root() / "datasets" / "x" / "corpus.jsonl"
    with pytest.raises(RuntimeError) as e:
        _download_one("http://fake/corpus.jsonl", wrong_sha, dest, "x corpus")
    assert "checksum mismatch" in str(e.value)
    assert not dest.exists()


def test_gzip_download_is_decompressed(tmp_path, monkeypatch):
    inner = b'{"id": "1", "text": "zipped"}\n'
    gz = gzip.compress(inner)
    _serve(monkeypatch, gz)
    sha = hashlib.sha256(inner).hexdigest()  # checksum is over decompressed content
    dest = cache_root() / "datasets" / "x" / "corpus.jsonl"
    out = _download_one("http://fake/corpus.jsonl.gz", sha, dest, "x corpus")
    assert out.read_bytes() == inner
