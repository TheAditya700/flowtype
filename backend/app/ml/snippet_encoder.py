from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings

# Load the model only once when the module is imported
model = SentenceTransformer(settings.embedding_model)

def get_snippet_embedding(text: str) -> np.ndarray:
    """
    Encodes a text snippet into a vector embedding.
    """
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype('float32')
