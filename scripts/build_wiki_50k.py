"""Build a local 50K-doc Wikipedia corpus for a Flat-vs-HNSW benchmark.

Source: wikimedia/wikipedia, config 20231101.simple (Simple English Wikipedia,
CC BY-SA 4.0). Streamed so we stop after collecting enough articles without
downloading the whole set into memory.

Output (git-ignored, local-only):
  examples/data/wiki-50k/corpus.jsonl   {id, text}  ~50,000 docs
  examples/data/wiki-50k/queries.jsonl  {id, text}  200 real article titles

Each corpus text = "<title>. <lead>" truncated to ~600 chars, so embedding stays
fast (short passages) while retaining real semantics. Recall is measured vs exact
Flat search on the SAME embeddings, so no relevance labels are needed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset

N_DOCS = 50_000
N_QUERIES = 200
MAX_CHARS = 600
MIN_BODY_CHARS = 200

OUT_DIR = Path(__file__).resolve().parents[1] / "examples" / "data" / "wiki-50k"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.simple", split="train", streaming=True
    )

    corpus_path = OUT_DIR / "corpus.jsonl"
    kept = 0
    titles: list[str] = []
    with open(corpus_path, "w", encoding="utf-8") as out:
        for row in ds:
            title = clean(row["title"])
            body = clean(row["text"])
            if len(body) < MIN_BODY_CHARS or not title:
                continue
            text = f"{title}. {body}"[:MAX_CHARS]
            out.write(json.dumps({"id": str(row["id"]), "text": text}) + "\n")
            titles.append(title)
            kept += 1
            if kept % 5000 == 0:
                print(f"  {kept} docs written")
            if kept >= N_DOCS:
                break

    # Queries = article titles, evenly spread across the corpus (real short queries).
    step = max(1, len(titles) // N_QUERIES)
    picks = titles[::step][:N_QUERIES]
    queries_path = OUT_DIR / "queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as out:
        for i, t in enumerate(picks):
            out.write(json.dumps({"id": str(i), "text": t}) + "\n")

    print(f"Done: {kept} docs -> {corpus_path}")
    print(f"      {len(picks)} queries -> {queries_path}")


if __name__ == "__main__":
    main()
