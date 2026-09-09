"""Dense Semantic Retrieval Vector Store for Historical Support Resolutions."""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np


class ResolutionRetrievalIndex:
    """Retrieval index mapping customer queries to verified historical brand resolutions

    strictly populated from the Training split.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self._encoder = None
        self.corpus: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None

    @property
    def encoder(self):
        """Lazy-loaded SentenceTransformer encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name, device=self.device)
        return self._encoder

    def build_index(self, pairs: List[Dict], batch_size: int = 64) -> "ResolutionRetrievalIndex":
        """Encodes customer queries from training pairs and constructs vector index."""
        self.corpus = pairs
        queries = [p["customer_text"] for p in pairs]

        # Compute normalized dense embeddings (L2 norm = 1.0, so dot product = cosine similarity)
        self.embeddings = self.encoder.encode(
            queries,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        try:
            import faiss
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
        except ImportError:
            self.faiss_index = None

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
        assert self.embeddings is not None, "Index must be built before search"
        query_vec = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        if self.faiss_index is not None and not intent_filter:
            # Global FAISS search
            scores, indices = self.faiss_index.search(query_vec, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.corpus):
                    results.append((self.corpus[idx], float(score)))
            return results

        # Fallback / Intent-filtered search via vector dot products
        scores = np.dot(self.embeddings, query_vec.T).squeeze(-1)  # shape (N,)

        candidate_indices = []
        for i, doc in enumerate(self.corpus):
            if intent_filter and doc.get("intent") != intent_filter:
                continue
            candidate_indices.append(i)

        if not candidate_indices and intent_filter:
            # Fall back to global if filtered pool is empty
            candidate_indices = list(range(len(self.corpus)))

        # Sort candidate indices by score descending
        candidate_scores = [(idx, scores[idx]) for idx in candidate_indices]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        top_results = []
        for idx, score in candidate_scores[:top_k]:
            top_results.append((self.corpus[idx], float(score)))

        return top_results

    def save(self, index_dir: str):
        """Saves index embeddings, FAISS binary, and corpus metadata to disk."""
        out = Path(index_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "retrieval_metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, indent=2)

        np.save(out / "embeddings.npy", self.embeddings)

        if self.faiss_index is not None:
            try:
                import faiss
                faiss.write_index(self.faiss_index, str(out / "faiss.bin"))
            except Exception:
                pass

    @classmethod
    def load(cls, index_dir: str, device: str = "cpu") -> "ResolutionRetrievalIndex":
        """Loads index and metadata from disk."""
        inp = Path(index_dir)
        with open(inp / "retrieval_metadata.json", "r", encoding="utf-8") as f:
            corpus = json.load(f)

        embeddings = np.load(inp / "embeddings.npy")
        obj = cls(device=device)
        obj.corpus = corpus
        obj.embeddings = embeddings

        faiss_path = inp / "faiss.bin"
        if faiss_path.exists():
            try:
                import faiss
                obj.faiss_index = faiss.read_index(str(faiss_path))
            except Exception:
                obj.faiss_index = None

        return obj
