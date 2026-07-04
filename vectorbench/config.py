"""Pydantic v2 models for the experiment YAML config, plus a readable loader."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SyntheticConfig(_Strict):
    """Parameters for a locally generated synthetic vector corpus."""

    n_docs: int = Field(gt=0)
    n_queries: int = Field(gt=0)
    dim: int = Field(gt=0)


class RemoteConfig(_Strict):
    """A checksum-verified, downloaded-on-first-run corpus."""

    name: str
    url: str
    sha256: str
    queries_url: str
    queries_sha256: str


class DatasetConfig(_Strict):
    """Exactly one of three dataset forms: local files, synthetic, or remote."""

    corpus: Path | None = None
    queries: Path | None = None
    synthetic: SyntheticConfig | None = None
    remote: RemoteConfig | None = None

    @model_validator(mode="after")
    def _exactly_one_form(self) -> "DatasetConfig":
        local = self.corpus is not None or self.queries is not None
        forms = [local, self.synthetic is not None, self.remote is not None]
        n = sum(forms)
        if n == 0:
            raise ValueError(
                "dataset: specify exactly one form — local ('corpus' + 'queries'), "
                "'synthetic', or 'remote'"
            )
        if n > 1:
            raise ValueError(
                "dataset: specify exactly one form; found more than one of "
                "local / synthetic / remote"
            )
        if local:
            if self.corpus is None or self.queries is None:
                raise ValueError(
                    "dataset: local form requires both 'corpus' and 'queries'"
                )
            for label, p in (("corpus", self.corpus), ("queries", self.queries)):
                if not p.exists():
                    raise ValueError(f"dataset.{label}: file not found: {p}")
        return self

    @property
    def kind(self) -> str:
        if self.synthetic is not None:
            return "synthetic"
        if self.remote is not None:
            return "remote"
        return "local"


class EmbeddingConfig(_Strict):
    """The embedding model. v0.1 supports one model but the field is explicit."""

    model: str = "BAAI/bge-small-en-v1.5"


class HNSWConfig(_Strict):
    """HNSW build params (M, ef_construction fixed per experiment) + the efSearch sweep."""

    M: int = Field(default=32, gt=0)
    ef_construction: int = Field(default=200, gt=0)
    ef_search: list[int] = Field(default_factory=lambda: [16, 32, 64, 128, 256, 512])

    @field_validator("ef_search")
    @classmethod
    def _non_empty_positive(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("index.hnsw.ef_search: must be a non-empty list")
        if any(x <= 0 for x in v):
            raise ValueError("index.hnsw.ef_search: all values must be positive ints")
        return v


class IndexConfig(_Strict):
    """Index configuration. Flat is always the ground truth in v0.1."""

    flat: bool = True
    hnsw: HNSWConfig = Field(default_factory=HNSWConfig)

    @field_validator("flat")
    @classmethod
    def _flat_required(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "index.flat: must be true in v0.1 (Flat is the exact-search ground truth)"
            )
        return v


class EvaluationConfig(_Strict):
    """Evaluation knobs: k, warmup/min query counts, repetitions."""

    k: int = Field(default=10, ge=1)
    warmup_queries: int = Field(default=5, ge=0)
    min_queries: int = Field(default=200, ge=1)
    repetitions: int = Field(default=3, ge=1)


class OutputConfig(_Strict):
    """Where the run folder is written and what the report file is called."""

    dir: Path = Path("runs/")
    report_name: str = "experiment_report.html"


class ExperimentConfig(_Strict):
    """Top-level experiment definition parsed from a YAML file."""

    name: str
    dataset: DatasetConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    similarity: str = "cosine"
    index: IndexConfig = Field(default_factory=IndexConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    seed: int = 42

    @field_validator("similarity")
    @classmethod
    def _only_cosine(cls, v: str) -> str:
        if v != "cosine":
            raise ValueError(
                f"similarity: '{v}' is not supported in v0.1; "
                "only 'cosine' is supported; see docs/ROADMAP.md"
            )
        return v


class ConfigError(Exception):
    """Raised with a readable, single-issue message on invalid config."""


def _first_error_message(exc: ValidationError) -> str:
    """Render the first validation error as a one-line, field-named message."""
    err = exc.errors()[0]
    loc = ".".join(str(x) for x in err["loc"]) or "<root>"
    msg = err["msg"]
    # Custom ValueErrors already carry a full "field: message" string.
    if msg.startswith("Value error, "):
        return msg[len("Value error, ") :]
    return f"{loc}: {msg}"


def load_config(path: Path) -> ExperimentConfig:
    """Parse and validate a YAML config file, raising ConfigError with a readable message."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config file must contain a YAML mapping at the top level")
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_first_error_message(exc)) from exc
