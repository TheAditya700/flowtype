import numpy as np
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models.db_models import Snippet
from app.ml.vector_store import VectorStore
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_index():
    """
    Fetches all snippets from DB, gets existing embeddings, and builds a FAISS index.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    logger.info("Fetching all snippets from the database...")
    # Use processed_embedding for the index
    snippets = session.query(Snippet).filter(Snippet.processed_embedding.isnot(None)).all()
    session.close()
    
    if not snippets:
        logger.warning("No snippets with processed_embeddings found. Run snippet vectorization first.")
        return

    logger.info(f"Found {len(snippets)} snippets. Building index...")
    
    embeddings = []
    metadata = []
    
    for snippet in snippets:
        # Use processed_embedding
        emb_list = snippet.processed_embedding
        if not emb_list:
            continue
            
        embeddings.append(emb_list)
        metadata.append({
            "id": str(snippet.id),
            "words": snippet.words,
            "difficulty": snippet.difficulty_score,
            "embedding": emb_list # Store PROCESSED embedding for Two-Tower ranking
        })

    if not embeddings:
        logger.warning("No valid embeddings found.")
        return

    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    logger.info(f"Embeddings shape: {embeddings_np.shape}. Building and saving FAISS index...")
    
    vector_store = VectorStore()
    # Initialize index with correct dimension
    import faiss
    vector_store.index = faiss.IndexFlatL2(embeddings_np.shape[1])
    
    vector_store.add(embeddings_np, metadata)
    vector_store.save()
    
    logger.info(f"FAISS index and metadata saved to {vector_store.index_path} and {vector_store.metadata_path}")

if __name__ == "__main__":
    build_index()