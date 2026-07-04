# Bundled small corpus — SciFact

`corpus.jsonl` (5,183 docs) and `queries.jsonl` (300 queries) are the **SciFact** corpus and
its test-set claims, as packaged by the [BEIR benchmark](https://github.com/beir-cellar/beir).
SciFact is a dataset of expert-written scientific claims paired with evidence abstracts drawn
from a corpus of research-paper abstracts.

## Source

- **Dataset:** SciFact (Wadden et al., 2020), via the BEIR distribution.
- **Downloaded from:** `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip`
- **Upstream:** <https://github.com/allenai/scifact> · <https://huggingface.co/datasets/BeIR/scifact>

## Licence

**CC BY-SA 4.0** (Creative Commons Attribution-ShareAlike 4.0 International), as tagged on the
BEIR SciFact dataset card. The upstream `allenai/scifact` release licenses claims/annotations
under CC BY 4.0 and abstracts under ODC-By 1.0; the BEIR packaging used here is CC BY-SA 4.0.

Because CC BY-SA is share-alike, these **data files** (and any adaptation of them) are
redistributed under CC BY-SA 4.0 with attribution to the SciFact authors. This obligation
attaches to the data only — the VectorBench **code** remains under the repository's MIT
licence. See `docs/DATASET_LICENSES.md` for the full determination (Ticket T0).

## Attribution

> Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020).
> *Fact or Fiction: Verifying Scientific Claims.* EMNLP 2020.

If you redistribute these files, keep this attribution and the CC BY-SA 4.0 notice.

## Extraction

Produced from the BEIR `scifact.zip` with no manual editing:

- **`corpus.jsonl`** — every corpus document as `{"id": <_id>, "text": "<title> <abstract>"}`
  (title and abstract concatenated with a single space). 5,183 documents.
- **`queries.jsonl`** — the 300 claims referenced by `qrels/test.tsv` (the canonical SciFact
  *test* query set), as `{"id": <_id>, "text": <claim>}`.

No relevance judgments (qrels) are bundled: VectorBench measures Recall@k against **exact
search** (the Flat index), so labelled relevance is not needed.
