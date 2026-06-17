"""Corpus loading, embedding and similarity search.

The corpus is loaded and embedded exactly once at startup; queries are a
single embedding plus one dot product against the matrix, which is fast
enough at this scale without any vector store.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("backend.retrieval")

# Resolve relative to the repo root (parent of backend/), not the current
# working directory, so the server can be launched from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# One-line change later: either edit this default or set CORPUS_PATH.
CORPUS_PATH = Path(
    os.environ.get("CORPUS_PATH", PROJECT_ROOT / "data" / "corpus.json")
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SNIPPET_LENGTH = 200

REQUIRED_FIELDS = ("id", "title", "parent_title", "source", "text")

# Raw chunks scored before dedup. Larger pool means the true best chunk per
# article is more likely to appear before we start dropping duplicates.
CANDIDATE_POOL = 30

# Sibling-expansion settings for /answer context.
# The top SIBLING_TOP_N committed articles get their representative chunk
# plus up to MAX_SIBLINGS_PER_ARTICLE additional chunks from the same article,
# so a fact sitting in a neighbouring chunk is still reachable by the generator.
SIBLING_TOP_N = 2
MAX_SIBLINGS_PER_ARTICLE = 2
SIBLING_TOTAL_CAP = 4  # absolute ceiling; stays inside SmolLM2's context window


class Retriever:
    """Holds the corpus, its embeddings and the embedding model."""

    def __init__(self, corpus_path: Path = CORPUS_PATH) -> None:
        self.corpus_path = corpus_path
        self.entries = self._load_corpus(corpus_path)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # normalize_embeddings=True lets us use a plain dot product as
        # cosine similarity at query time.
        texts = [entry["text"] for entry in self.entries]
        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        self.by_id = {entry["id"]: entry for entry in self.entries}
        # Chunks grouped by parent_title, sorted by id so sibling order is stable.
        self._siblings: dict[str, list[dict]] = {}
        for entry in sorted(self.entries, key=lambda e: e["id"]):
            self._siblings.setdefault(entry["parent_title"], []).append(entry)
        logger.info(
            "Loaded corpus %s with %d entries", corpus_path, len(self.entries)
        )

    @staticmethod
    def _load_corpus(path: Path) -> list[dict]:
        if not path.is_file():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError(f"Corpus {path} must be a non-empty JSON array")
        for i, entry in enumerate(data):
            for field in REQUIRED_FIELDS:
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    raise ValueError(
                        f"Corpus entry {i} is missing or has empty field '{field}'"
                    )
        return data

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the top-k results for the query, one chunk per article, best first.

        Scores CANDIDATE_POOL raw chunks, keeps only the highest-scoring chunk
        per parent_title (dedup), then returns the top-k distinct articles.
        top1_score and top2_score therefore reflect the inter-article gap, so
        the confidence margin is meaningful even on a corpus with many chunks
        per article.
        """
        query_vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ query_vec

        pool_size = min(CANDIDATE_POOL, len(self.entries))
        pool_indices = np.argsort(scores)[::-1][:pool_size]

        # One representative chunk per article: keep the highest scorer.
        best_per_article: dict[str, tuple[int, float]] = {}
        for idx in pool_indices:
            parent = self.entries[idx]["parent_title"]
            score = float(scores[idx])
            if parent not in best_per_article or score > best_per_article[parent][1]:
                best_per_article[parent] = (int(idx), score)

        ranked = sorted(best_per_article.values(), key=lambda t: t[1], reverse=True)
        k = min(k, len(ranked))

        results = []
        for idx, score in ranked[:k]:
            entry = self.entries[idx]
            results.append(
                {
                    "doc_id": entry["id"],
                    "title": entry["title"],
                    "parent_title": entry["parent_title"],
                    "score": round(score, 4),
                    "snippet": entry["text"][:SNIPPET_LENGTH],
                }
            )
        return results

    def get_entries(self, doc_ids: list[str]) -> list[dict]:
        """Return corpus entries for known ids, silently skipping unknown ones."""
        return [self.by_id[d] for d in doc_ids if d in self.by_id]

    def get_entries_with_siblings(self, doc_ids: list[str]) -> list[dict]:
        """Return entries for doc_ids, expanded with sibling chunks for top articles.

        For the top SIBLING_TOP_N articles, also pulls up to MAX_SIBLINGS_PER_ARTICLE
        additional chunks from the same parent_title. This ensures a fact that sits in
        a different chunk from the top-ranked one is still present for the generator.
        Total is capped at SIBLING_TOTAL_CAP.
        """
        collected: list[dict] = []
        seen_ids: set[str] = set()

        for rank, doc_id in enumerate(doc_ids):
            if doc_id not in self.by_id:
                continue
            hit = self.by_id[doc_id]

            if hit["id"] not in seen_ids:
                collected.append(hit)
                seen_ids.add(hit["id"])

            if rank < SIBLING_TOP_N:
                siblings = self._siblings.get(hit["parent_title"], [])
                added = 0
                for sib in siblings:
                    if added >= MAX_SIBLINGS_PER_ARTICLE:
                        break
                    if sib["id"] not in seen_ids:
                        collected.append(sib)
                        seen_ids.add(sib["id"])
                        added += 1

            if len(collected) >= SIBLING_TOTAL_CAP:
                break

        return collected[:SIBLING_TOTAL_CAP]
