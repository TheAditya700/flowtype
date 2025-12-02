import numpy as np
import torch
from sqlalchemy.orm import Session
from app.database import engine
from app.models.db_models import Snippet
from app.ml.difficulty_features import compute_difficulty_features
from app.ml.snippet_tower import get_snippet_model


# ----- FIXED FEATURE ORDER (FAISS stable) -----
FEATURE_KEYS = [
    "vowel_ratio",
    "consonant_ratio",
    "digit_ratio",
    "punct_ratio",
    "space_ratio",

    "same_finger_ratio",
    "hand_alt_ratio",
    "row_change_ratio",
    "pinky_ratio",
    "ring_ratio",
    "pinky_runs",

    "flow_segments",
    "longest_flow_segment",
    "avg_flow_segment_length",

    "left_to_right_ratio",
    "right_to_left_ratio",
    "direction_changes",
    "direction_change_ratio",

    "max_char_run",
    "avg_char_run",
    "double_letter_count",
    "triple_letter_count",

    "bigram_log_freq_avg",
    "bigram_log_freq_min",
    "trigram_log_freq_avg",
    "trigram_log_freq_min",
    "rare_ngram_ratio",

    "word_log_freq_avg",
    "word_log_freq_min",
    "word_log_freq_max",
]


# ------------------------------------------------------------
# Single-snippet feature extractor (online usage)
# ------------------------------------------------------------
def vectorize_snippet(text: str) -> np.ndarray:
    feats = compute_difficulty_features(text)
    return np.array([float(feats.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)


# ------------------------------------------------------------
# Full normalization utility
# ------------------------------------------------------------
def normalize_matrix(X: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Normalizes across columns.
    """
    if method == "zscore":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        return (X - mean) / std

    elif method == "minmax":
        mn = X.min(axis=0, keepdims=True)
        mx = X.max(axis=0, keepdims=True)
        return (X - mn) / (mx - mn + 1e-6)

    else:
        raise ValueError("Unknown normalization method")


# ------------------------------------------------------------
# Vectorize + Normalize + Save Back to DB
# ------------------------------------------------------------
def vectorize_and_update_all_snippets(normalize: bool = True, method: str = "zscore"):
    """
    Loads all snippets, converts raw feature dicts → vectors,
    normalizes them, passes them through Snippet Tower, 
    and writes both normalized features + processed embeddings back into DB.
    """
    session = Session(bind=engine)
    model = get_snippet_model()

    # Step 1: Load all snippet rows
    rows = session.query(Snippet).all()

    if not rows:
        print("No snippets found!")
        session.close()
        return

    # Step 2: Extract vectors in consistent order
    matrix = []
    for snip in rows:
        vec = np.array([float(snip.features.get(k, 0.0)) for k in FEATURE_KEYS], dtype=np.float32)
        matrix.append(vec)

    X = np.vstack(matrix)  # [N, 30]

    # Step 3: Normalize entire matrix
    if normalize:
        X_norm = normalize_matrix(X, method=method)
    else:
        X_norm = X.copy()
        
    # Step 4: Pass through Snippet Tower
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    with torch.no_grad():
        processed_embeddings = model(X_tensor).numpy()

    # Step 5: Save back to DB
    for i, snip in enumerate(rows):
        norm_vec = X_norm[i].tolist()
        proc_vec = processed_embeddings[i].tolist()
        
        snip.normalized_features = norm_vec
        snip.embedding = norm_vec         # Raw normalized features (legacy/debug)
        snip.processed_embedding = proc_vec # Output of Snippet Tower (used for search)

    session.commit()
    session.close()

    print(f"✓ Updated {len(rows)} snippets with normalized features + processed embeddings.")

if __name__ == "__main__":
    vectorize_and_update_all_snippets(
        normalize=True,
        method="zscore"
    )
