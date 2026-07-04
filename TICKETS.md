# VectorBench MVP — TICKETS.md

Companion to PLAN.md (v3). Tickets are ordered by dependency. Each ticket is self-contained enough for Claude Code to execute against, but read PLAN.md §4 (architecture + key design decisions) before starting any ticket. Tickets marked **[HUMAN]** require Shreyas, not Claude Code.

Conventions for all tickets:
- Python ≥3.10, type hints throughout, no bare `except:`.
- Every module gets a short docstring stating its single responsibility.
- No abstractions beyond what the ticket specifies. No plugin interfaces, no ABCs, no "for future extensibility" code. If tempted, add a note to docs/ROADMAP.md instead.

---

## T0 — Dataset license verification **[HUMAN — BLOCKING for T5b, T5c]**

**Goal:** Confirm redistributable licensing for the two real corpora before any data is committed or a download URL is pinned.

**Tasks:**
1. Check SciFact's actual license in the BEIR release / original paper repo. Record finding.
2. Check FiQA (or BEIR NQ subset) license for the 50K tier. Record finding.
3. If either is unclear or non-commercial: fall back to building both corpora from a Wikipedia dump subset (CC BY-SA). Small tier: ~5K intro paragraphs + 300 paraphrase queries. 50K tier: ~50–100K paragraphs + 300+ queries. Host the 50K files as a GitHub release asset on the vectorbench repo.
4. Write `examples/data/small/README.md` documenting source, license, extraction method.

**Acceptance criteria:**
- [ ] A written license determination for both corpora exists (one paragraph each is fine).
- [ ] Bundled small corpus is CC-BY / CC BY-SA / equivalent — redistributable in an MIT-licensed repo with attribution.
- [ ] 50K corpus has a stable public URL and its SHA256 is recorded for pinning in `benchmark-50k.yaml`.

---

## T1 — Repo scaffold + pyproject

**Goal:** Installable package skeleton.

**Tasks:**
1. Create the repo structure exactly as PLAN.md §4.
2. `pyproject.toml`: package name `vectorbench`, version `0.1.0`, deps from PLAN §8, console script `vectorbench = vectorbench.cli:app`.
3. Empty-but-importing modules with docstrings. `LICENSE` (MIT), stub `README.md`, stub `docs/ROADMAP.md`.
4. `.gitignore`: `runs/`, `__pycache__`, `.pytest_cache`, `~/.cache` never applies but exclude any local `*.faiss`/`*.npy` artifacts.

**Acceptance criteria:**
- [ ] `pip install -e .` succeeds in a fresh venv.
- [ ] `vectorbench --help` prints Typer help.
- [ ] `python -c "import vectorbench"` works.

---

## T2 — Config models (`config.py`)

**Goal:** Pydantic v2 models implementing PLAN §6 schema exactly.

**Tasks:**
1. Models: `ExperimentConfig`, `DatasetConfig` (discriminated: `local` | `synthetic` | `remote`), `EmbeddingConfig`, `IndexConfig`, `HNSWConfig`, `EvaluationConfig`, `OutputConfig`.
2. `model_config = ConfigDict(extra="forbid")` everywhere.
3. Validators: exactly one dataset form; `similarity == "cosine"` else `ValueError("only 'cosine' is supported in v0.1; see docs/ROADMAP.md")`; `repetitions >= 1`; `k >= 1`; `ef_search` non-empty list of positive ints; local file paths must exist at load time with a clear message naming the missing path.
4. `load_config(path: Path) -> ExperimentConfig` — parses YAML, raises with readable messages.

**Acceptance criteria:**
- [ ] All `test_config.py` cases in T10 pass.
- [ ] Error messages name the offending field and value; no raw Pydantic wall-of-text reaches the CLI user (CLI formats first error line only unless `--debug`).

---

## T3 — Hashing (`hashing.py`)

**Goal:** Content-aware config hash per PLAN §4.

**Tasks:**
1. `canonicalize(config: ExperimentConfig) -> str`: deterministic serialization (sorted keys, normalized whitespace) of the *validated model*, not the raw YAML text.
2. `file_sha256(path) -> str` streaming, memory-safe.
3. `experiment_hash(config, corpus_hash, queries_hash) -> str` = sha256 over canonical config + corpus hash + queries hash + embedding model name + `faiss.__version__`. For synthetic datasets, substitute the synthetic params string for file hashes.
4. `short(hash) -> str` = first 8 chars.

**Acceptance criteria:**
- [ ] `test_hashing.py` cases pass (same inputs → same hash; corpus content change → different hash; model name change → different hash; YAML key reordering → same hash).

---

## T4 — Metrics (`metrics.py`)

**Goal:** Correct, tested metric functions. No pipeline knowledge.

**Tasks:**
1. `recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> float` — mean over queries of |retrieved[:k] ∩ truth[:k]| / k. Handles: result rows shorter than k (pad with -1, never counts), duplicate IDs in retrieved (count once).
2. `latency_stats(latencies_ns: np.ndarray) -> LatencyStats` dataclass: p50_ms, p95_ms, iqr_ms, n.
3. `aggregate_reps(recalls: list[float]) -> tuple[mean, std]` — std is population std when n==1 returns std 0.0, not NaN.

**Acceptance criteria:**
- [ ] All `test_metrics.py` cases in T10 pass, including the tie and empty-result edge cases.
- [ ] Pure functions: numpy in, numbers out, no I/O, no FAISS imports.

---

## T5a — Dataset: local loader + synthetic generator (`dataset.py` part 1)

**Goal:** Load JSONL corpora and generate the synthetic corpus.

**Tasks:**
1. `load_jsonl(path) -> Dataset` where `Dataset = (ids: list[str], texts: list[str])`. Validates every line has `id` and `text`; on malformed line, raise with line number.
2. `generate_synthetic(n_docs, n_queries, dim, seed) -> SyntheticDataset` — random normal vectors, L2-normalized, cached as `.npy` in `~/.cache/vectorbench/synthetic_<n>_<dim>_<seed>/`. Returns vectors directly (bypasses embedding stage).
3. Deterministic: same seed → identical arrays (assert in smoke test).

**Acceptance criteria:**
- [ ] Malformed JSONL produces `"corpus.jsonl line 42: missing 'text' field"`-style errors.
- [ ] Synthetic generation is idempotent and cached (second call reads from cache; verify via mtime or a log line).

## T5b — Bundled small corpus **[depends on T0]**

**Goal:** Commit the small real corpus.

**Tasks:**
1. Produce `examples/data/small/corpus.jsonl` (~5K docs) and `queries.jsonl` (~300 queries) from the T0-approved source.
2. `README.md` in that directory: source, license, extraction script or notes.
3. Keep total under ~20MB.

**Acceptance criteria:**
- [ ] `load_jsonl` reads both files without error.
- [ ] License documentation present and consistent with T0's determination.

## T5c — Remote dataset downloader (`dataset.py` part 2) **[depends on T0]**

**Goal:** One-time checksum-verified download for the 50K corpus.

**Tasks:**
1. `download_dataset(name, url, sha256, dest_dir) -> Path` — streaming GET with `requests`, `tqdm` progress bar, write to temp file, verify SHA256, atomic rename into `~/.cache/vectorbench/datasets/<name>/`. Gunzip if `.gz`.
2. On checksum mismatch: delete temp file, raise `"checksum mismatch for <name>: expected …, got … — the file may be corrupted or the URL changed"`.
3. On network failure: raise `"this config requires a one-time download (~<size>MB); check your connection or use examples/scifact-small.yaml for a fully offline run"`.
4. If cached and checksum-valid, skip download entirely.

**Acceptance criteria:**
- [ ] Unit test with mocked/`file://` response covers: fresh download, cached hit, checksum failure. No network in CI.
- [ ] Interrupted download never leaves a corrupt file in the cache dir (temp + atomic rename).

---

## T6 — Embedding (`embedding.py`)

**Goal:** BGE-small embedding with normalization and disk cache.

**Tasks:**
1. `embed_texts(texts, model_name, cache_key) -> np.ndarray (float32, L2-normalized)` using `sentence_transformers.SentenceTransformer`, batch size 64, `normalize_embeddings=True` (verify it actually L2-normalizes; if not, apply `faiss.normalize_L2` after).
2. Disk cache: `~/.cache/vectorbench/embeddings/<model_slug>/<cache_key>.npy` where `cache_key` = corpus content hash. Load if present.
3. Progress bar for corpora >1K docs.
4. Assert output norms ≈ 1.0 (tolerance 1e-3) before returning — this is the invariant the whole L2-equals-cosine design rests on.

**Acceptance criteria:**
- [ ] Second run with same corpus + model hits the cache (verifiable via log line, covered in smoke test).
- [ ] Returned array is float32, shape (n, 384) for bge-small, all norms ≈ 1.

---

## T7 — Index (`index.py`)

**Goal:** FAISS Flat + HNSW with the build-once contract and single-threaded reproducibility.

**Tasks:**
1. Module-level: call `faiss.omp_set_num_threads(1)` once (runner calls an `init_faiss()` from here).
2. `build_flat(vectors) -> FlatIndex` — `IndexFlatL2`, wraps search to return (ids, latencies_ns per query). IDs mapped via an explicit id array (positional index → doc id).
3. `build_hnsw(vectors, ids, M, ef_construction, insertion_order: np.ndarray) -> HNSWIndex` — adds vectors in `insertion_order`; keeps the order→id mapping so search results return *doc ids*, not positional indices. Measure and store build time.
4. `HNSWIndex.set_ef_search(v)` mutates `index.hnsw.efSearch` in place — the ONLY way the sweep changes the index. No rebuild path exists in the API surface.
5. `search(query_vectors, k) -> (ids: (n,k) array, latencies_ns: (n,) array)` — per-query timing with `perf_counter_ns`, one query at a time (no batching — per-query latency is the metric).
6. `index_size_mb(index) -> float` via `faiss.write_index` to a temp file, stat, delete.

**Acceptance criteria:**
- [ ] Two builds with different insertion orders on the same vectors produce different graphs (different search results for at least one query at low efSearch) — asserted in smoke test.
- [ ] Two builds with the SAME insertion order produce identical search results (single-threaded determinism) — asserted in smoke test.
- [ ] Returned ids are doc ids, verified against Flat's ids for an easy query.
- [ ] No public rebuild-per-efSearch API exists.

---

## T8 — Runner (`runner.py`)

**Goal:** Orchestration implementing PLAN §4 data flow exactly.

**Tasks:**
1. `run_experiment(config: ExperimentConfig) -> ExperimentResult`:
   - init faiss (single-threaded), compute hashes, resolve dataset (local/synthetic/remote), embed (or use synthetic vectors directly), collect machine info.
   - Flat: build, warmup (5 queries, discarded), timed search over all queries (enforce `min_queries` — if the query set is smaller, raise a clear config error at validation time instead, in T2), store ground-truth top-K + latencies.
   - Loop `rep in range(repetitions)`: `rng = np.random.default_rng(config.seed + rep)`; `order = rng.permutation(n)`; build HNSW once; loop `ef in ef_search`: set efSearch, warmup 5, timed search, store (rep, ef) → ids + latencies.
   - Compute metrics: per-(ef) recall mean±std across reps; latency stats pooled per (ef) across reps' per-query timings.
2. `ExperimentResult` dataclass: everything the report needs + raw per-rep numbers, machine info, versions, hashes, timestamps. Serializable to the `results.json` schema (`schema_version: 1`).
3. Structured progress logging: one line per major stage, one per (rep, ef) cell.

**Acceptance criteria:**
- [ ] Smoke test (T10) passes end-to-end.
- [ ] `results.json` contains raw per-rep recalls and per-query latency arrays are summarized (store percentiles, not 500 raw numbers per cell — keep the file <1MB).
- [ ] HNSW build happens exactly `repetitions` times regardless of sweep length (assert via counter in test).

---

## T9 — Report (`report.py`)

**Goal:** Self-contained HTML report per PLAN §7, plus results.json writer.

**Tasks:**
1. Jinja2 template (inline in the package or `templates/experiment_report.html.j2`): header (name + experiment type + hash), config block, metric-definition callout (+ synthetic disclaimer when applicable), summary table, Pareto chart div, variance note, machine info block, footer with full hash + reproduce command.
2. Plotly figure: scatter, x = p50 latency ms (log axis), y = Recall@10. HNSW points: mean markers, `error_y` = recall std, `error_x` = latency IQR/2, text labels = efSearch values. Flat: horizontal line at y=1.0 with annotation. `include_plotlyjs='inline'`, `full_html=False`, embedded into the Jinja template.
3. Clean, minimal CSS (system font stack, one accent color, table borders). No external assets whatsoever — the HTML must render offline.
4. `write_outputs(result, out_dir)` → registry-ready run folder `runs/<timestamp>_<shorthash>/` containing:
   - `experiment_report.html` — the artifact is called the **Experiment Report** in all user-facing copy (file name, HTML `<title>`, CLI output line). Never bare "report" or "benchmark report".
   - `results.json`
   - `config.yaml` — the *resolved* config as actually run (dump the validated Pydantic model, post-defaults), not a copy of the input file
   - `metadata.json` — `experiment_type: "flat-vs-hnsw"` (hardcoded string in v0.1 — no dispatch, no abstraction), name, full hash, timestamp (ISO 8601), duration_s, dataset summary (name/type, n_docs, n_queries), machine info, vectorbench version + dependency versions. Small and flat; this is the future `vectorbench list` index source.
   The experiment type also appears in the report header next to the experiment name.

**Acceptance criteria:**
- [ ] Report renders in a browser with no network (verify no `http`/`https` references in the HTML except inside the config echo).
- [ ] Flat row has no ± values; HNSW rows do.
- [ ] Synthetic runs show the disclaimer; real runs don't.
- [ ] File opens correctly when double-clicked from Finder/Explorer (no relative asset paths).
- [ ] Run folder contains all four files; `config.yaml` in the folder re-runs successfully via `vectorbench run` and produces the same config hash (round-trip check, asserted in smoke test).

---

## T10 — Tests

**Goal:** The four test files from PLAN §9 + downloader test from T5c.

**Tasks:** Implement exactly the cases listed in PLAN §9. Smoke test uses `synthetic: {n_docs: 200, n_queries: 20, dim: 32}`, `repetitions: 2`, `ef_search: [16]`, and asserts: outputs exist, Flat self-recall == 1.0, HNSW recall ∈ [0,1], reps used different insertion orders, same-order rebuilds are deterministic, build counter == repetitions.

**Acceptance criteria:**
- [ ] `pytest` green in <2 min on CPU, no network.
- [ ] No test depends on the bundled real corpus or the remote download (those are exercised manually).

---

## T11 — CLI polish (`cli.py`)

**Goal:** The final UX surface.

**Tasks:**
1. `vectorbench run CONFIG [--debug]`. Prints: config name, experiment type, short hash, dataset summary, then stage progress, then final line: `Experiment report: runs/<dir>/experiment_report.html`.
2. Top-level exception handler: one line `Error during experiment <shorthash>: <message>`, exit 1. `--debug` → full traceback.
3. `vectorbench --version`.

**Acceptance criteria:**
- [ ] A deliberately broken config (missing file) produces exactly one readable error line without `--debug`.
- [ ] Happy path ends with the report path as the last line.

---

## T12 — Docs: README + ROADMAP

**Goal:** The public face.

**Tasks:**
1. `README.md`: one-paragraph pitch — lead with **"Design, run, compare, and visualize retrieval experiments on your own data."** Benchmarking appears as the flagship use case ("v0.1 ships the first experiment type: Flat vs HNSW with efSearch sweeps and error bars"), never as the project identity. Then: install; the three example commands with what each is for; dataset table from PLAN §5; screenshot placeholder for the 50K Pareto chart (added in T14); honest platform note (macOS/Linux, Windows via WSL2); metric honesty note; link to ROADMAP. The word "benchmark" should not appear in the first two sentences.
2. `docs/ROADMAP.md`: the original vision document, reorganized with a status table at top — Phase 0–1 marked SHIPPED (v0.1.0), everything else FUTURE. Keep the vision's ambition intact; it's a feature. Three required sections:
   - **Experiment Types:** frame every future capability as a new experiment type on the same engine — `flat-vs-hnsw` (shipped) → embedding comparison → database comparison → chunking comparison → hybrid search → RAG evaluation. Note that v0.1 already records `experiment_type` in every run's metadata, so the registry is type-aware from day one.
   - **Experiment Registry:** `vectorbench list` (reads metadata.json across runs/), `vectorbench compare <run1> <run2>` (merged Pareto chart + config diff), `vectorbench export` (bundle a run folder for sharing) — noting no migration needed. Frame as the "MLflow for retrieval" direction.
   - **Version evolution:** v0.1 Flat/HNSW + Experiment Report + run folders → v0.2 IVF, better charts, `compare` → v0.3 Qdrant/Chroma/LanceDB → v0.4 embedding comparison → v0.5 chunking experiments → v0.6 hybrid search → v1.0 retrieval experimentation platform. Explicitly state the no-plugin rule: "abstractions are extracted after the second implementation exists, never before."
3. GitHub Action: `pytest` + `vectorbench run examples/quickstart.yaml` (synthetic) as smoke on ubuntu-latest and macos-latest.

**Acceptance criteria:**
- [ ] A newcomer can go from clone to first report using only the README.
- [ ] ROADMAP status table exists; no future feature is described as existing.

---

## T13 — Full-scale validation run **[HUMAN]**

**Goal:** First real measured run of `benchmark-50k.yaml` on Shreyas's machine.

**Tasks:**
1. Run it. Check: total time within budget (first run <10 min, cached <3 min), chart shows visible Flat/HNSW gap, knee identifiable, error bars sane (recall std < 0.02 typically; if larger, investigate).
2. Sanity-check absolute numbers against intuition (Flat p50 at 50K/384-dim should be low tens of ms single-threaded; HNSW low single-digit ms at moderate efSearch). If wildly off, debug before publishing anything.
3. Adjust `ef_search` sweep values if the knee falls outside the range or all points cluster.

**Acceptance criteria:**
- [ ] A report you'd be willing to screenshot publicly.
- [ ] The knee finding is written down as one sentence with real numbers.

---

## T14 — Ship: v0.1.0 + LinkedIn post **[HUMAN]**

**Tasks:**
1. Add the real chart screenshot to README. Tag `v0.1.0`. Upload the 50K dataset as a release asset if self-hosting (per T0).
2. LinkedIn post: headline = the T13 knee finding ("On <corpus>, HNSW's efSearch knee is at ~N: beyond it, Mx latency for <0.X% recall gain"). Hero image = the Pareto chart. Body: 3–4 sentences on why pipeline-level benchmarking with error bars matters, link to repo. Log as O-1A evidence (original tool + public technical finding).

**Acceptance criteria:**
- [ ] Repo public, tagged, CI green.
- [ ] Post published with the chart and a specific numeric claim.

---

## Build order summary

```
T0 (human, parallel) ──────────────┐
T1 → T2 → T3 → T4 → T5a → T6 → T7 → T8 → T9 → T10 → T11 → T12
                    T5b, T5c (after T0) ──┘ (needed by T13, not by T8–T12)
T13 (human) → T14 (human)
```

Everything from T1–T12 runs on synthetic + small corpus; the 50K path is only load-bearing for T13/T14. So Claude Code can build the entire tool while T0's licensing check happens in parallel.
