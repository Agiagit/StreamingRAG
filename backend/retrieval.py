"""Corpus loading, embedding and similarity search.

The corpus is loaded and embedded exactly once at startup; queries are a
single embedding plus one dot product against the matrix, which is fast
enough at this scale (a few hundred entries) without any vector store.
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
    os.environ.get("CORPUS_PATH", PROJECT_ROOT / "data" / "sample_corpus.json")
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SNIPPET_LENGTH = 200

REQUIRED_FIELDS = ("id", "title", "parent_title", "source", "text")


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
        """Return the top-k entries for the query, best first.

        Each result carries the frozen response fields plus the raw entry id
        so callers can look up full texts for generation.
        """
        query_vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ query_vec
        k = min(k, len(self.entries))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            entry = self.entries[idx]
            results.append(
                {
                    "doc_id": entry["id"],
                    "title": entry["title"],
                    "parent_title": entry["parent_title"],
                    "score": round(float(scores[idx]), 4),
                    "snippet": entry["text"][:SNIPPET_LENGTH],
                }
            )
        return results

    def get_entries(self, doc_ids: list[str]) -> list[dict]:
        """Return corpus entries for known ids, silently skipping unknown ones
        so a stale id from the client does not break answer generation."""
        return [self.by_id[d] for d in doc_ids if d in self.by_id]
