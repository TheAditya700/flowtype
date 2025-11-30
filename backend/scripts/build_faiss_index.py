import numpy as np
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models.db_models import Snippet
from app.ml.snippet_encoder import get_snippet_embedding
from app.ml.vector_store import VectorStore
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_index():
    """
    Fetches all snippets from DB, computes embeddings, and builds a FAISS index.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    logger.info("Fetching all snippets from the database...")
    snippets = session.query(Snippet).all()
    session.close()
    
    if not snippets:
        logger.warning("No snippets found in the database. Run load_corpus.py first.")
        return

    logger.info(f"Found {len(snippets)} snippets. Computing embeddings...")
    
    embeddings = []
    metadata = []
    
    for snippet in snippets:
        embedding = get_snippet_embedding(snippet.words)
        embeddings.append(embedding)
        metadata.append({
            "id": str(snippet.id),
            "words": snippet.words,
            "difficulty": snippet.difficulty_score
        })

    embeddings_np = np.array(embeddings)
    
    logger.info("Embeddings computed. Building and saving FAISS index...")
    
    vector_store = VectorStore()
    vector_store.add(embeddings_np, metadata)
    vector_store.save()
    
    logger.info(f"FAISS index and metadata saved to {vector_store.index_path} and {vector_store.metadata_path}")

if __name__ == "__main__":
    build_index()
