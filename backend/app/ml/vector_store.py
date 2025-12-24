import faiss
import numpy as np
import json
from pathlib import Path
from app.config import settings
from app.utils.s3_data import get_env_data_bucket, download_to_temp, upload_bytes


class VectorStore:
    def __init__(self):
        self.index = None
        self.metadata = []

        # Try to load from S3 data bucket for current env
        try:
            self.load()
        except Exception:
            # Create empty index if load fails
            self.index = faiss.IndexFlatL2(settings.embedding_dim)

    def load(self):
        """Load FAISS index and metadata from S3 (env bucket)."""
        bucket = get_env_data_bucket(settings.env)
        # Download index to temp file for FAISS
        idx_tmp_path = download_to_temp(bucket, settings.faiss_index_key)
        self.index = faiss.read_index(str(idx_tmp_path))
        # Load metadata JSON
        meta_tmp_path = download_to_temp(bucket, settings.snippet_metadata_key)
        with open(meta_tmp_path) as f:
            self.metadata = json.load(f)

    def save(self):
        """Persist FAISS index and metadata to S3 (env bucket)."""
        bucket = get_env_data_bucket(settings.env)
        # Serialize FAISS index to bytes
        # Use temp path since serialize_index may not be available
        tmp = Path("/tmp/faiss_index.bin")
        faiss.write_index(self.index, str(tmp))
        body = tmp.read_bytes()
        upload_bytes(bucket, settings.faiss_index_key, body)
        # Upload metadata JSON
        upload_bytes(
            bucket,
            settings.snippet_metadata_key,
            json.dumps(self.metadata).encode("utf-8"),
            content_type="application/json",
        )

    def add(self, embeddings: np.ndarray, metadata: list[dict]):
        """Add vectors and metadata to index"""
        self.index.add(embeddings)  # type: ignore
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, k: int = 50) -> list[dict]:
        """
        Search for similar snippets using FAISS (L2 distance) in shared embedding space.
        Returns list of {snippet_id, words, distance}
        """
        if not self.index or self.index.ntotal == 0:
            return []

        # Search in FAISS
        # We request k directly since we aren't filtering anymore
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype("float32"), k
        )  # pyright: ignore[reportCallIssue]

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            snippet = self.metadata[idx]
            results.append({**snippet, "distance": float(dist)})

        return results

    def get_embedding_by_id(self, snippet_id: str) -> np.ndarray | None:
        """Reconstruct a snippet embedding from FAISS by snippet id."""
        try:
            # metadata list index aligns with FAISS index ids
            for idx, meta in enumerate(self.metadata):
                if str(meta.get("id")) == str(snippet_id):
                    return self.index.reconstruct(idx)
        except Exception:
            return None
        return None
