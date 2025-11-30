import numpy as np
from app.models.schema import UserState
from app.config import settings

def get_user_embedding(state: UserState) -> np.ndarray:
    """
    Encodes user state into a vector embedding.
    This is a simplified placeholder. A real implementation would be more complex.
    """
    # Normalize features (very roughly)
    wpm_norm = state.rollingWpm / 150.0
    acc_norm = state.rollingAccuracy
    back_norm = state.backspaceRate / 5.0
    diff_norm = state.currentDifficulty / 10.0
    
    # Create a feature vector
    feature_vector = np.array([
        wpm_norm,
        acc_norm,
        back_norm,
        diff_norm
    ], dtype=np.float32)
    
    # This is a simple projection. A real model might use a learned MLP.
    # We need to project this into the same dimension as the snippet embeddings.
    # For now, we'll just pad with zeros.
    embedding = np.zeros(settings.embedding_dim, dtype=np.float32)
    
    # Simple projection: repeat the feature vector to fill the space
    len_features = len(feature_vector)
    for i in range(0, settings.embedding_dim, len_features):
        end_pos = min(i + len_features, settings.embedding_dim)
        embedding[i:end_pos] = feature_vector[:end_pos-i]
        
    return embedding
