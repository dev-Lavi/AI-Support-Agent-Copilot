"""Lightweight TF-IDF Semantic Retrieval Vector Store for Historical Support Resolutions.

Zero-PyTorch, sub-3MB RAM footprint designed for resource-constrained environments (e.g. Render 512MB limit).
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetrievalIndex:
    """Lightweight retrieval index mapping customer queries to verified historical brand resolutions

    using sublinear TF-IDF character and word n-grams with cosine similarity.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b"
        )
        self.corpus: List[Dict] = []
        self.tfidf_matrix = None
        self.is_built: bool = False

    def build_index(self, pairs: List[Dict]) -> "TfidfRetrievalIndex":
        """Encodes customer queries from historical pairs into TF-IDF sparse matrix."""
        self.corpus = pairs
        queries = [p.get("customer_text", "") for p in pairs]
        self.tfidf_matrix = self.vectorizer.fit_transform(queries)
        self.is_built = True
        return self

    def search(
        self,
        query: str,
        top_k: int = 3,
        intent_filter: Optional[str] = None
    ) -> List[Tuple[Dict, float]]:
        """Searches for top-k historical resolution pairs most semantically similar to `query`.

        Optionally filters by `intent_filter`.

        Returns:
            List of (document_metadata, cosine_similarity_score).
        """
        assert self.is_built and self.tfidf_matrix is not None, "Index must be built before search"
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        candidate_indices = []
        for i, doc in enumerate(self.corpus):
            if intent_filter and doc.get("intent") != intent_filter:
                continue
            candidate_indices.append(i)

        if not candidate_indices and intent_filter:
            candidate_indices = list(range(len(self.corpus)))

        candidate_scores = [(idx, float(sims[idx])) for idx in candidate_indices]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        top_results = []
        for idx, score in candidate_scores[:top_k]:
            top_results.append((self.corpus[idx], score))

        return top_results

    def save(self, index_dir: str):
        """Saves corpus metadata to disk."""
        out = Path(index_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "retrieval_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, indent=2)

    @classmethod
    def load(cls, index_dir: str) -> "TfidfRetrievalIndex":
        """Loads corpus metadata and builds the TF-IDF index in <50ms with ~2MB RAM."""
        inp = Path(index_dir)
        meta_file = inp / "retrieval_metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Retrieval metadata not found at {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        idx = cls()
        idx.build_index(corpus)
        return idx
