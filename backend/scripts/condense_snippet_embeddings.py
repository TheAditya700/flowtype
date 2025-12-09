import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models.db_models import Snippet
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "app/ml/snippet_pca_16.pkl"

def condense_embeddings():
    """
    1. Fetches raw 32-dim features from Snippet.features.
    2. Scales them (StandardScaler).
    3. Runs PCA to reduce to 16 dimensions.
    4. Updates Snippet.embedding (16-dim) and Snippet.difficulty_score (derived from PC1).
    5. Saves the Scaler and PCA model.
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    
    logger.info("Fetching snippets...")
    snippets = session.query(Snippet).all()
    
    if not snippets:
        logger.warning("No snippets found in database.")
        session.close()
        return

    # 1. Vectorize Features
    logger.info(f"Found {len(snippets)} snippets. Vectorizing features...")
    
    # Sort keys to ensure consistency
    if not snippets[0].features or not isinstance(snippets[0].features, dict):
        logger.error("First snippet has invalid features. Aborting.")
        session.close()
        return
        
    feature_keys = sorted(snippets[0].features.keys())
    logger.info(f"Using {len(feature_keys)} features: {feature_keys}")
    
    X_list = []
    valid_snippets = []
    
    for s in snippets:
        if isinstance(s.features, dict):
            vec = [float(s.features.get(k, 0.0)) for k in feature_keys]
            X_list.append(vec)
            valid_snippets.append(s)
            
    X = np.array(X_list)
    logger.info(f"Feature Matrix Shape: {X.shape}")
    
    # 2. Scale
    logger.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. PCA (16 components)
    n_components = 16
    logger.info(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled) # (N, 16)
    
    logger.info(f"Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    # 4. Calculate Difficulty (PC1 scaled 1-10)
    # We use PC1 as a proxy for difficulty/complexity.
    # We normalize it to [1, 10].
    pc1 = X_pca[:, 0]
    min_s = pc1.min()
    max_s = pc1.max()
    
    # Map min->1, max->10
    difficulty_scores = 1.0 + 9.0 * (pc1 - min_s) / (max_s - min_s + 1e-6)

    # 5. Update Database
    logger.info("Updating database with embeddings and difficulty scores...")
    
    for i, s in enumerate(valid_snippets):
        s.embedding = X_pca[i].tolist()
        s.difficulty_score = float(difficulty_scores[i])
        # We can also store normalized features if needed, but user asked for embeddings
        # s.normalized_features = X_scaled[i].tolist() 
    
    session.commit()
    session.close()
    
    # 6. Save Model
    logger.info(f"Saving model to {MODEL_PATH}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    model_data = {
        "scaler": scaler,
        "pca": pca,
        "feature_keys": feature_keys,
        "diff_min": min_s,
        "diff_max": max_s
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
        
    logger.info("Done.")

if __name__ == "__main__":
    condense_embeddings()
