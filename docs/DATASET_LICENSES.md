# Dataset licence determination (Ticket T0)

This document records the licence determination that T0 requires **before** any corpus is
committed to the repo or a download URL is pinned. Licences were verified against primary
sources (dataset LICENSE files and dataset cards), not secondary summaries — an early web
summary claiming SciFact is "CC BY-NC 2.0" was **incorrect** and is contradicted by the
official `allenai/scifact` LICENSE.md.

## Findings

| Corpus | Verified licence | Source | Redistributable in an MIT repo? |
|---|---|---|---|
| **SciFact** (BeIR/scifact, 5,183 docs) | **CC BY-SA 4.0** (BEIR packaging). Upstream `allenai/scifact`: claims/annotations **CC BY 4.0**, abstracts **ODC-By 1.0** | allenai/scifact `LICENSE.md`; `BeIR/scifact` dataset card | ✅ Yes — attribution + share-alike on the data file |
| **NQ** (BeIR/nq) | **CC BY-SA 4.0** | `BeIR/nq` dataset card; BEIR paper (Table, NQ = CC BY-SA) | ✅ Yes — attribution + share-alike |
| **FiQA-2018** | Non-commercial / academic use only | BEIR paper; `BeIR/fiqa` card | ❌ **No — do not bundle or redistribute** |
| **Wikipedia** (fallback) | CC BY-SA 4.0 | Wikimedia licensing policy | ✅ Yes — unambiguous |

## Determination

- **Small tier** (`examples/data/small/`, T5b): **SciFact** (BeIR/scifact), 5,183 docs +
  ~300 test queries. CC BY-SA 4.0. Matches PLAN §5's ~5K-doc / ~300-query target and its
  first-choice candidate. Replaces the synthetic placeholder currently in that directory.
- **50K tier** (`benchmark-50k.yaml`, T5c): a **50K-document subset of NQ** (BeIR/nq) plus
  300+ of its natural-language questions as queries. CC BY-SA 4.0.
  - Note: because recall is measured **against exact search (Flat) on the same embeddings**,
    no relevance judgments (qrels) are needed — only real document and query *text*. Any
    50K real docs + 300+ real queries suffice; NQ is chosen for realistic query phrasing.
  - Hosting is a GitHub release asset on the vectorbench repo (per T0/PLAN §5). The single
    `corpus.jsonl.gz` + `queries.jsonl.gz` SHA256s get pinned into `benchmark-50k.yaml`.
- **FiQA is excluded** — its non-commercial restriction is incompatible with redistribution
  in this repo.

## Share-alike scope (important, not alarming)

CC BY-SA 4.0 is a *share-alike* licence. Its obligations attach to **the data files we
redistribute and to adaptations of that data** (e.g. a derived `corpus.jsonl`): those must
carry a CC BY-SA 4.0 notice and attribution to the original dataset. It does **not** relicense
the VectorBench **code**, which remains MIT — share-alike does not reach independent software
that merely reads or processes the data. This is the same separation BEIR itself ships under,
and it is captured by PLAN §13.4 ("dataset licences documented separately per corpus").

Each bundled/downloaded corpus therefore ships with its own attribution + licence notice in a
sibling `README.md` (small tier) or release notes (50K tier); the repo's top-level `LICENSE`
(MIT) governs the code only.

## Status of T0 acceptance criteria

- [x] Written licence determination for both corpora — this document.
- [x] Bundled small corpus is CC BY-SA 4.0 (redistributable with attribution) — SciFact.
- [ ] 50K corpus has a stable public URL and pinned SHA256 — **pending**: requires building
      the 50K NQ subset and uploading it as a GitHub release asset (needs the repo pushed).
