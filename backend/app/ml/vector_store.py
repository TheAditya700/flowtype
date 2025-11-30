import faiss
import numpy as np
import json
from pathlib import Path
from app.config import settings

class VectorStore:
    def __init__(self):
        self.index_path = Path(settings.faiss_index_path)
        self.metadata_path = Path(settings.snippet_metadata_path)
        self.index = None
        self.metadata = []
        
        if self.index_path.exists() and self.metadata_path.exists():
            self.load()
        else:
            # Create empty index
            self.index = faiss.IndexFlatL2(settings.embedding_dim)
    
    def load(self):
        """Load FAISS index and metadata from disk"""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path) as f:
            self.metadata = json.load(f)
    
    def save(self):
        """Save FAISS index and metadata to disk"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def add(self, embeddings: np.ndarray, metadata: list[dict]):
        """Add vectors and metadata to index"""
        self.index.add(embeddings)
        self.metadata.extend(metadata)
    
    def search(
        self, 
        query_vector: np.ndarray, 
        k: int = 50,
        difficulty_min: float = 1.0,
        difficulty_max: float = 10.0
    ) -> list[dict]:
        """
        Search for similar snippets with difficulty filtering
        Returns list of {snippet_id, words, difficulty, distance}
        """
        # Search in FAISS
        distances, indices = self.index.search(
            query_vector.reshape(1, -1).astype('float32'), 
            k * 3
        )
        
        # Filter by difficulty and return top k
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            snippet = self.metadata[idx]
            if difficulty_min <= snippet['difficulty'] <= difficulty_max:
                results.append({
                    **snippet,
                    'distance': float(dist)
                })
            
            if len(results) >= k:
                break
        
        return results
