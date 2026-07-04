# VectorBench Roadmap

VectorBench is an **experimentation platform for retrieval pipelines**. Benchmarking is its
first use case, not its identity. The organizing idea is that every run *is* an experiment of
a declared **type**, recorded in that run's `metadata.json`. New capabilities ship as new
experiment types on the same engine and the same run-folder format — no migrations.

## Status

| Phase | Capability | Status |
|---|---|---|
| 0–1 | `flat-vs-hnsw` experiment type: FAISS Flat vs HNSW, `efSearch` sweep, error bars | **SHIPPED (v0.1.0)** |
| 0–1 | Self-contained Experiment Report + registry-ready run folders | **SHIPPED (v0.1.0)** |
| 0–1 | Content-aware config hashing, single-threaded reproducibility | **SHIPPED (v0.1.0)** |
| 2 | IVF index, richer charts, `vectorbench compare` | Future |
| 3 | Second vector database (Qdrant / Chroma / LanceDB) | Future |
| 4 | Embedding-model comparison experiment type | Future |
| 5 | Chunking-strategy experiment type | Future |
| 6 | Hybrid (dense + sparse) search experiment type | Future |
| 1.0 | Full retrieval experimentation platform + Experiment Registry | Future |

Nothing below the shipped rows exists yet. It describes intended direction, not current
behaviour.

## Experiment Types

Every future capability is framed as a new experiment type on the same engine:

- **`flat-vs-hnsw`** *(shipped)* — approximate vs exact search, `efSearch` sweep.
- **embedding comparison** — same corpus and queries, N embedding models, compare recall
  and latency of the retrieval they produce.
- **database comparison** — same vectors, N vector databases, compare build/query/recall.
- **chunking comparison** — same documents, N chunking strategies, compare downstream recall.
- **hybrid search** — dense + sparse fusion vs dense-only.
- **RAG evaluation** — end-to-end answer quality as a function of retrieval config.

Because v0.1 already records `experiment_type` in every run's `metadata.json`, the registry
below is type-aware from day one and can filter by type without touching old runs.

## Experiment Registry ("MLflow for retrieval")

The v0.1 run folder is the on-disk substrate for a registry that reads `metadata.json`
across `runs/`:

- **`vectorbench list`** — index all runs by type, name, date, dataset, hash.
- **`vectorbench compare <run1> <run2>`** — a merged Pareto chart plus a config diff.
- **`vectorbench export <run>`** — bundle a run folder for sharing.

No migration is needed: every run written by v0.1 is already registry-ready.

## Version evolution

- **v0.1** — Flat/HNSW, Experiment Report, run folders *(shipped)*.
- **v0.2** — IVF, better charts, `compare`.
- **v0.3** — Qdrant / Chroma / LanceDB.
- **v0.4** — embedding-model comparison.
- **v0.5** — chunking experiments.
- **v0.6** — hybrid search.
- **v1.0** — retrieval experimentation platform with the full Experiment Registry.

## The no-plugin rule

Abstractions are extracted **after** the second implementation exists, never before. v0.1
hardcodes FAISS, one embedding model, and one experiment type on purpose. When a second
database or experiment type lands, the shared interface is factored out from two real cases —
not guessed at from one. If you are tempted to add a plugin interface "for extensibility,"
add a line here instead.
