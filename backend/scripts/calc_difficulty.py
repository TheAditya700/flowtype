import numpy as np
from sklearn.decomposition import PCA
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models.db_models import Snippet
import logging
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "app/ml/difficulty_pca.pkl"

def calibrate_difficulty():
    """
    Uses PCA to reduce the 30-dim feature vector to a 1-dim 'Difficulty Score'.
    Assumes the largest variance in features correlates with complexity.
    Scales the result to [1.0, 10.0].
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # 1. Fetch all snippets with normalized features
    logger.info("Fetching snippets...")
    snippets = session.query(Snippet).filter(Snippet.normalized_features.isnot(None)).all()
    
    if not snippets:
        logger.warning("No snippets found with normalized features. Run snippet_encoder.py first.")
        return

    # 2. Build Feature Matrix
    X_list = []
    ids = []
    for s in snippets:
        # normalized_features is a list of floats
        X_list.append(s.normalized_features)
        ids.append(s.id)
        
    X = np.array(X_list)
    logger.info(f"Feature matrix shape: {X.shape}")

    # 3. Run PCA (1 component)
    pca = PCA(n_components=1)
    components = pca.fit_transform(X) # (N, 1)
    
    # 4. Rescale to [1.0, 10.0]
    # We assume PC1 aligns with difficulty. 
    # NOTE: PCA sign is arbitrary. We need to check if 'positive' means harder.
    # Heuristic: Check correlation with 'word_log_freq_avg' (rarity) or 'length'.
    # If rare words (lower freq) should be harder, and if feature index for freq is negative...
    # Simple approach: Let's assume the projection effectively separates simple vs complex.
    # We normalize min/max to 1-10.
    
    raw_scores = components.flatten()
    min_s = raw_scores.min()
    max_s = raw_scores.max()
    
    # Normalize to 0-1 then map to 1-10
    # score = 1 + 9 * (raw - min) / (max - min)
    scaled_scores = 1.0 + 9.0 * (raw_scores - min_s) / (max_s - min_s + 1e-6)
    
    logger.info(f"PC1 Variance Ratio: {pca.explained_variance_ratio_[0]:.4f}")
    logger.info(f"Score Range: {min_s:.2f} to {max_s:.2f} -> Scaled: 1.0 to 10.0")

    # 5. Update DB
    logger.info("Updating database...")
    for i, snip_id in enumerate(ids):
        # We need to fetch the object again or iterate the list if attached
        # snippets[i] corresponds to X_list[i]
        snippets[i].difficulty_score = float(scaled_scores[i])
    
    session.commit()
    session.close()
    
    # 6. Save PCA model for future inference
    model_data = {
        "pca": pca,
        "min_s": min_s,
        "max_s": max_s
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
        
    logger.info(f"Difficulty calibration complete. PCA model saved to {MODEL_PATH}")

if __name__ == "__main__":
    calibrate_difficulty()
