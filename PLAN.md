# VectorBench MVP — PLAN.md (v3)

## 1. Purpose

**Positioning:** VectorBench is an experimentation platform for retrieval pipelines — *design, run, compare, and visualize retrieval experiments on your own data.* Benchmarking is the first use case of the experimentation engine, not the project's identity. This framing is used consistently in the README, ROADMAP, and report copy, and it's what lets rerankers, chunking, hybrid search, and RAG evaluation join later without the project outgrowing its name.

Build the smallest possible working core: a CLI that runs a reproducible retrieval experiment comparing FAISS Flat and HNSW on real text datasets with a local embedding model, sweeps `efSearch` with multiple repetitions for error bars, and emits a self-contained HTML report with a Recall-vs-Latency Pareto chart.

The full vision (plugin system, dashboard, multiple databases, IVF, MRR/NDCG, RAG eval, etc.) ships as `docs/ROADMAP.md` — clearly labeled as future work. The MVP is a weekend build, not a platform.

Success = one command, one config file, one HTML report, one LinkedIn post with a defensible headline claim.

---

## 2. Scope

### In scope (MVP)

- Single CLI entry point: `vectorbench run <config.yaml>` (plus `--debug`)
- One local embedding model: `BAAI/bge-small-en-v1.5` (CPU-friendly, 384-dim)
- One vector backend: FAISS
- Two index types: `IndexFlatL2` (ground truth) and `IndexHNSWFlat` (native L2)
- Embeddings L2-normalized at write time → L2 distance ranks identically to cosine; single code path
- JSONL dataset loader (`{id, text}` per line) + queries JSONL
- **Three bundled example configs (three distinct roles):**
  - `examples/quickstart.yaml` → synthetic vectors. Role: smoke test / CI. Zero network, <60s. Semantics are meaningless (disclaimer included). **Revised during build from 50K → 10K:** HNSW builds single-threaded (for reproducibility), and near-equidistant random vectors are HNSW's worst case for build cost, so 50K synthetic took ~5.5 min single-threaded — blowing the CI-smoke budget. 10K keeps a full multi-point Pareto with an obvious knee at ~32s cold. The scale is cosmetic for this config (the vectors are noise).
  - `examples/scifact-small.yaml` → bundled ~5K-doc real corpus. Role: fast real-embedding run. Latency spread will be small at this scale (disclaimer included).
  - `examples/benchmark-50k.yaml` → ~50–100K-doc real corpus, downloaded on first run. Role: **the LinkedIn chart.** Visible Flat-vs-HNSW latency gap, real semantics.
- Simple dataset downloader for the 50K corpus: single HTTPS fetch with checksum verification, cached in `~/.cache/vectorbench/datasets/`
- Metrics: Recall@10 vs exact search, p50 / p95 / IQR latency, index build time, index size on disk
- Sweep dimension: `efSearch` (list of values)
- Repetitions with **insertion-order shuffling** (see §4) for honest error bars; single-threaded builds for reproducibility
- Output: single self-contained HTML report (Plotly inline, error bars) + `results.json`
- Content-aware config hash (YAML + corpus + queries + model + FAISS version)
- Clean failure handling: one-line error + config hash; `--debug` for traceback

### Out of scope (goes to ROADMAP.md)

- Plugin architecture (hardcoded imports for MVP)
- FastAPI backend, React frontend, SQLite storage
- IVF, DiskANN, ScaNN, PQ, OPQ
- Any second vector database; any second embedding model; remote embedding APIs
- Chunking, rerankers, hybrid search, RAG eval
- MRR, NDCG, Precision@K, Hit Rate, Coverage, Diversity
- Embedding explorer, HNSW graph traversal animation
- PDF / Markdown reports; leaderboards across runs; experiment comparison UI
- Experiment Registry **commands** (`vectorbench list`, `compare`, `export`) — but the registry-ready run-folder format ships in v0.1 (see §7)
- Auto-tuning, cost estimation
- Windows testing (README: "Tested on macOS and Linux. Windows via WSL2.")

---

## 3. Non-goals

- Not a database. Not benchmarking-as-a-service. Not a research paper.
- Not competing with ann-benchmarks on breadth, and not positioned as a benchmark suite at all. The differentiator is *reproducible retrieval experimentation on your own data* with honest error bars, in a polished single-file HTML report. "Which config wins" is one question the engine answers, not the product.
- Not building for extensibility yet. Hardcode. Extract abstractions only when a second implementation exists.

---

## 4. Architecture

Single Python package, flat and boring.

```
vectorbench/
  __init__.py
  cli.py              # Typer entry point, --debug flag
  config.py           # Pydantic models for YAML config
  dataset.py          # JSONL loader + synthetic generator + downloader (checksum-verified)
  embedding.py        # BGE-small wrapper; L2-normalizes on output; disk cache
  index.py            # FAISS Flat + HNSW builders; build-once contract; single-threaded
  runner.py           # Orchestrates: load → embed → build → reps × sweep → measure
  metrics.py          # recall_at_k, latency percentiles + IQR
  report.py           # HTML template + Plotly figure with error bars
  hashing.py          # Content-aware config hash
  machine_info.py     # CPU model, RAM, Python/FAISS/torch versions
examples/
  quickstart.yaml     # synthetic 10K (smoke/CI)
  scifact-small.yaml  # bundled small real corpus
  benchmark-50k.yaml  # downloaded 50-100K real corpus (LinkedIn config)
  data/
    small/
      corpus.jsonl
      queries.jsonl
      README.md       # source, license, extraction notes
tests/
  test_metrics.py
  test_config.py
  test_hashing.py
  test_runner_smoke.py
docs/
  ROADMAP.md
README.md
pyproject.toml
LICENSE               # MIT
```

### Data flow

1. `cli.py` parses YAML → `ExperimentConfig` (Pydantic). Top-level try/except: on failure print `Error during experiment <short_hash>: <message>`, exit 1. `--debug` re-raises.
2. `runner.py`:
   a. Compute content-aware config hash. Set `faiss.omp_set_num_threads(1)`.
   b. `dataset.py` loads corpus + queries — from local JSONL, synthetic generation, or checksum-verified download (cached).
   c. `embedding.py` embeds corpus + queries, L2-normalizes, caches to disk keyed by `hash(model_name + corpus_content_hash)`.
   d. `index.py` builds `IndexFlatL2` once → all queries → ground-truth top-K + Flat latency distribution.
   e. For each repetition `r` in `0..repetitions`: **shuffle vector insertion order with seed `base_seed + r`**, build HNSW once from the shuffled order (ID mapping preserved via `IndexIDMap` or explicit ID array), then for each `efSearch` in the sweep: mutate `index.hnsw.efSearch` in place, run all queries, record per-query latencies + top-K. **Build once per rep, search many.**
   f. `metrics.py` computes Recall@10 per (efSearch, rep) vs ground truth; latency percentiles + IQR.
3. `report.py` renders `experiment_report.html` + writes `results.json`, `config.yaml` (resolved), `metadata.json`.

### Key design decisions

- **Flat as ground truth eliminates the labeled-data problem.** Recall@10 = overlap between HNSW top-10 and Flat top-10 on the same embeddings. Labeled everywhere as `Recall@10 (vs exact search)` with the explanatory sentence: "Measures how faithfully HNSW approximates exact search on the same embeddings; does not measure retrieval quality against human relevance judgments."

- **L2 on normalized vectors, everywhere.** BGE is trained for cosine; unit-normalizing makes L2 ranking identical to cosine ranking. FAISS HNSW is L2-native — avoids `METRIC_INNER_PRODUCT` cross-version fragility. Config's `similarity:` accepts only `cosine` in v0.1.

- **Repetition mechanism = insertion-order shuffling, single-threaded builds.** FAISS HNSW exposes no per-build construction seed; single-threaded construction is deterministic given insertion order, and real-world variance comes largely from insertion order (and thread interleaving). So: each repetition shuffles the insertion order with `base_seed + rep_index`, builds single-threaded (`omp_set_num_threads(1)`). This yields genuinely different graphs per rep, is fully controllable and reproducible, and is honest about the variance source. Documented in the report ("variance across repetitions reflects sensitivity to insertion order"). Cost: slower builds; acceptable at ≤100K vectors (a few minutes total). Doc/query IDs are preserved through shuffling so recall computation is unaffected.

- **Build-once contract.** `build_hnsw(params, insertion_order) → index` is called once per repetition. The efSearch sweep mutates `index.hnsw.efSearch` on the same object and re-queries. Build time measured once per build, never per sweep point.

- **Error bar sources are explicit.** Recall error bars: across-repetition std. Latency error bars: within-run per-query IQR. Flat is deterministic → single row, no error bars — visually communicating its ground-truth role.

- **Latency methodology.** Single-threaded, warm, per-query `time.perf_counter_ns`. 5 warmup queries discarded. Minimum 200 timed queries per config. Report p50 / p95 / IQR. Machine-info block: "laptop-grade measurements; background load not controlled."

- **Synthetic corpus disclaimer.** Normalized random high-dim vectors are near-equidistant; nearest neighbors are essentially arbitrary and HNSW recall may look anomalous. The quickstart config's description and its report carry a one-line note: "Synthetic random vectors — for smoke-testing the pipeline, not for drawing conclusions."

- **Embedding cache on disk.** Keyed by model name + corpus content SHA. Re-runs skip embedding.

- **Self-contained HTML.** `include_plotlyjs='inline'`. One file to email/screenshot/gist.

- **Content-aware config hash.** `sha256(canonical_yaml + corpus_content_hash + queries_content_hash + model_name + faiss_version)`. Short hash in header, full in footer + results.json.

---

## 5. Datasets

Three tiers, three roles:

| Config | Corpus | Size | Source | Role |
|---|---|---|---|---|
| `quickstart.yaml` | Synthetic random vectors | 10K vecs / 500 queries* | Generated locally | Smoke test, CI. ~30s single-threaded, zero network. |
| `scifact-small.yaml` | Small real corpus | ~5K docs / ~300 queries | Bundled in repo (~15MB) | Fast real-embedding run. Latency spread small at this scale (noted). |
| `benchmark-50k.yaml` | Real corpus | 50–100K docs / 300+ queries | **Downloaded on first run**, checksum-verified, cached | **The LinkedIn chart.** Visible latency gap + real semantics. |

\* Revised from 50K → 10K during the build to keep the CI-smoke role's <60s budget (50K synthetic HNSW builds took ~5.5 min single-threaded — the near-equidistant-vector worst case). See the `quickstart.yaml` header.

**⚠ BLOCKING pre-build verification (goes in TICKETS as Ticket 0):** confirm the license of every bundled/downloaded corpus *before* any data is committed or a download URL is hardcoded. BEIR datasets have mixed licenses; several are non-commercial. Requirements: bundled small corpus must be redistributable (CC-BY or equivalent); downloaded corpus must be from a stable public URL with clear licensing. **Safe fallback for both: Wikipedia-derived paragraph corpora (CC BY-SA), which are unambiguous.** Candidates to check in order: SciFact (small tier), FiQA or a BEIR NQ subset (50K tier); if licensing is unclear, build both from a Wikipedia dump subset and self-host the 50K file (GitHub release asset works — free, versioned, checksummable).

Downloader requirements: single HTTPS GET, SHA256 verified against a hash pinned in the config, progress bar, cached in `~/.cache/vectorbench/datasets/<name>/`, clear error message on network failure ("this config requires a one-time ~60MB download; use scifact-small.yaml for a fully offline run").

---

## 6. Config schema

```yaml
name: hnsw_efsearch_sweep

dataset:
  # exactly one of the three forms:
  corpus: examples/data/small/corpus.jsonl     # local files
  queries: examples/data/small/queries.jsonl
  # synthetic:
  #   n_docs: 50000
  #   n_queries: 500
  #   dim: 384
  # remote:
  #   name: benchmark-50k
  #   url: https://.../corpus.jsonl.gz
  #   sha256: <pinned>
  #   queries_url: https://.../queries.jsonl.gz
  #   queries_sha256: <pinned>

embedding:
  model: BAAI/bge-small-en-v1.5

similarity: cosine    # only 'cosine' in v0.1; implemented as L2 on normalized vectors

index:
  flat: true          # always true in MVP; ground truth
  hnsw:
    M: 32
    ef_construction: 200
    ef_search: [16, 32, 64, 128, 256, 512]

evaluation:
  k: 10
  warmup_queries: 5
  min_queries: 200
  repetitions: 3

output:
  dir: runs/
  report_name: experiment_report.html

seed: 42
```

Pydantic validation: unknown fields raise; exactly one dataset form; `similarity == "cosine"` (others → "not supported in v0.1; see ROADMAP"); `repetitions >= 1`; missing files → clear message.

---

## 7. Experiment Report contents

**Naming convention:** the artifact is called the **Experiment Report** everywhere — file name `experiment_report.html`, report `<title>`, README, CLI output ("Experiment report: runs/…"). Never just "report" or "benchmark report" in user-facing copy. Small thing; reinforces the positioning.

Single HTML file:

1. **Header:** experiment name, **experiment type** (`flat-vs-hnsw` in v0.1), timestamp, short config hash, machine info (CPU, RAM, Python/FAISS/torch versions).
2. **Config block:** rendered YAML.
3. **Metric definition callout:** the `Recall@10 (vs exact search)` explanation. For synthetic runs, the synthetic disclaimer.
4. **Summary table:** one row per config (Flat + each efSearch). Columns: label; Recall@10 mean ± std; p50 latency ms; p95 latency ms; latency IQR ms; build time s (mean across reps); index size MB. Flat row: single values, no ±.
5. **Pareto chart:** p50 latency (x, log) vs Recall@10 (y). HNSW points = mean across reps, vertical error bars = recall std, horizontal = latency IQR, each labeled with efSearch. Flat = horizontal reference line at Recall 1.0.
6. **Variance note:** one line — "variance across repetitions reflects HNSW's sensitivity to vector insertion order; builds are single-threaded for reproducibility."
7. **Machine info block:** laptop-grade disclaimer.
8. **Footer:** version, full config hash, reproduce command.

**Run folder (registry-ready substrate):** every run writes a self-describing folder —

```
runs/<timestamp>_<shorthash>/
  config.yaml              # resolved config as actually run (post-validation, post-defaults)
  results.json             # schema_version: 1, raw per-rep numbers
  experiment_report.html   # self-contained
  metadata.json            # experiment_type ("flat-vs-hnsw"), name, full hash, timestamp,
                           # duration_s, dataset summary, machine info, versions
```

**Experiment Types (conceptual model, one string in v0.1):** every run *is* an experiment of a declared type. v0.1 implements exactly one type — `flat-vs-hnsw` — hardcoded, no dispatch logic, no abstraction. But the type is recorded in `metadata.json` and shown in the report header from day one, so future types (embedding comparison, database comparison, chunking comparison, hybrid search) slot into the same registry and `vectorbench list` can filter by type without migrating old runs. The conceptual model ships now; the machinery ships when a second type exists.

This is the on-disk substrate for the future Experiment Registry (`vectorbench list / compare / export` — ROADMAP). The commands are out of scope for v0.1, but because every run is registry-ready from day one, they can be added later without migrating old runs.

---

## 8. Dependencies

`typer`, `pydantic` (v2), `pyyaml`, `sentence-transformers`, `faiss-cpu`, `numpy`, `plotly`, `jinja2`, `psutil`, `requests` (downloader), `tqdm` (progress), `pytest`.

No FastAPI, no React, no Polars, no SQLite.

---

## 9. Testing

- `test_metrics.py`: `recall_at_k` on hand-built cases (perfect, zero, partial, ties, K > result set, empty).
- `test_config.py`: valid YAML parses; missing/unknown fields raise; `similarity: dot` rejected; `repetitions: 0` rejected; two dataset forms simultaneously rejected.
- `test_hashing.py`: same YAML + same corpus → same hash; different corpus content → different hash; different model → different hash.
- `test_runner_smoke.py`: end-to-end on 200-doc / 20-query synthetic, `repetitions: 2`, one efSearch value; asserts report + results.json produced, Flat self-recall = 1.0, HNSW recall ∈ [0,1], and **the two reps produced different insertion orders** (sanity-check the shuffle actually happens).
- Downloader unit test with a local file:// or mocked response verifying checksum pass/fail paths. No network in CI.

Under 2 minutes on CPU.

---

## 10. Reproducibility guarantees

- Single-threaded FAISS builds. Insertion order fully determined by `base_seed + rep_index`.
- Content-aware config hash in report + results.json.
- FAISS, sentence-transformers, torch, Python, numpy versions in results.json.
- `results.json` schema versioned (`schema_version: 1`), includes raw per-rep numbers.
- Honest framing: repetition-based error bars are the reproducibility story, not a claim of bit-exactness across FAISS versions.

---

## 11. What ships in v0.1.0

- Working CLI: `pip install -e . && vectorbench run examples/scifact-small.yaml` produces a report offline; `benchmark-50k.yaml` produces the flagship chart after a one-time download.
- README: what it is, pitch, install, quickstart, screenshot of the 50K Pareto chart with error bars, dataset table, link to ROADMAP.
- ROADMAP.md: full vision with status table (MVP shipped vs future phases).
- MIT license. GitHub Action: pytest + synthetic quickstart smoke run.
- Not shipping: PyPI package (v0.2), contribution guide beyond stub, plugin docs, comparisons to other tools.

---

## 12. Success criteria

1. Fresh clone → `pip install -e .` → `scifact-small.yaml` completes in <3 min offline; `benchmark-50k.yaml` first run <10 min (download + embed, both cached), subsequent runs <3 min.
2. The 50K report shows a Pareto chart with a clearly visible latency gap between Flat and HNSW, error bars on every HNSW point, and an efSearch knee identifiable by eye.
3. Same config run twice → same config hash; per-rep recall differs, reported mean stable within one std.
4. README communicates the tool's purpose in 60 seconds.
5. The 50K Pareto chart is the hero image of a LinkedIn post whose headline claim is a specific efSearch knee finding (exact number measured, claim shape fixed in advance: "on <corpus>, the knee is at efSearch ≈ N — beyond it you pay Mx latency for <0.X% recall").

---

## 13. Decisions (locked)

1. Datasets: synthetic (CI — 10K, revised down from 50K for the <60s smoke budget) + bundled small real corpus + downloaded 50–100K real corpus. License verification is Ticket 0 and blocks data selection.
2. Repo name: `vectorbench`.
3. Sweep: efSearch only. M and efConstruction fixed per experiment.
4. License: MIT (code). Dataset licenses documented separately per corpus.
5. CLI: `vectorbench run <config.yaml>`, `--debug`. YAML is the source of truth.
6. Repetitions: insertion-order shuffling, single-threaded builds, default 3.
