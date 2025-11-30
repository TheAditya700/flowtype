"""
User embedding encoder. Requires numpy and other ML deps to be installed.
This is imported only when ML features are needed (e.g., snippets router).
"""
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from app.models.schema import UserState
from app.config import settings


def get_user_embedding(state: UserState):
    """
    Encodes user state into a vector embedding.
    Returns a dummy embedding if numpy is unavailable (e.g., in slim runtime image).
    """
    if not HAS_NUMPY:
        # Return a simple dummy embedding (all zeros) if numpy is not available
        # This allows the API to run without ML deps, though ranking will be basic
        return [0.0] * settings.embedding_dim
    
    # Normalize features to [0, 1] range
    wpm_norm = min(state.rollingWpm / 150.0, 1.0)  # Cap at 150 WPM
    acc_norm = state.rollingAccuracy  # Already 0-1
    back_norm = min(state.backspaceRate / 0.5, 1.0)  # Cap at 50% backspace rate
    diff_norm = state.currentDifficulty / 10.0  # 0-1 range
    
    # Create a base feature vector
    features = np.array([
        wpm_norm,
        acc_norm,
        back_norm,
        diff_norm
    ], dtype=np.float32)
    
    # Generate a pseudo-random projection using deterministic seeding
    # This creates a consistent but rich embedding from the 4 features
    seed = 42 
    rng = np.random.RandomState(seed)
    projection_matrix = rng.randn(4, settings.embedding_dim).astype('float32')
    projection_matrix = projection_matrix / np.linalg.norm(projection_matrix, axis=1, keepdims=True)
    
    # Project features to embedding space
    embedding = features @ projection_matrix
    
    # Normalize the resulting embedding
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    return embedding

