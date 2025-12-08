import torch
import numpy as np
from typing import List, Dict, Any, Optional
from app.ml.htom import get_htom_model
from app.ml.features import FeatureExtractor
from app.models.schema import UserState
from app.ml.user_features import UserFeatureExtractor

def rank_snippets(
    user_state: UserState,
    user_features_dict: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    target_difficulty: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Ranks candidate snippets using the HTOM model.
    
    1. Extract User Feature Vector (Global + Pooled History).
    2. Extract Snippet Feature Vector (Manual features).
    3. Pass through HTOM.
    4. Calculate score = w1*p_acc + w2*p_cons + w3*p_speed.
    """
    
    if not candidates:
        return []

    # --- 1. Prepare User Vector (57D) ---
    # Rehydrate the extractor to get the latest stats
    extractor = UserFeatureExtractor.from_dict(user_features_dict)
    # Incorporate the *current* session's rolling stats into the "short term history" logic if needed
    # For now, we just use the computed features which include history
    user_vec_np = extractor.compute_user_features() # (57,)
    user_tensor = torch.tensor(user_vec_np, dtype=torch.float32).unsqueeze(0) # (1, 57)

    # --- 2. Prepare Snippet Vectors ---
    # We need to compute manual features for each candidate
    snippet_feats_list = []
    for cand in candidates:
        text = cand.get('text') or " ".join(cand.get('words', []))
        feats = FeatureExtractor.compute_snippet_features(text) # (15,)
        snippet_feats_list.append(feats)
        
    snippet_tensor = torch.tensor(np.stack(snippet_feats_list), dtype=torch.float32) # (N, 15)
    
    # Expand user tensor to batch size
    batch_size = len(candidates)
    user_batch = user_tensor.expand(batch_size, -1) # (N, 57)
    
    # --- 3. Model Inference ---
    model = get_htom_model(user_dim=57, snippet_dim=15)
    
    with torch.no_grad():
        outputs = model(user_batch, snippet_tensor)
        p_acc = outputs['p_acc'].squeeze().numpy()
        p_cons = outputs['p_cons'].squeeze().numpy()
        p_speed = outputs['p_speed'].squeeze().numpy()
        
    # --- 4. Scoring ---
    # Weights for the hierarchy
    w_acc = 1.0
    w_cons = 0.5
    w_speed = 0.25
    
    # "Zone of Proximal Development" penalty based on difficulty
    # We want difficulty to be close to target (usually user's current level + small delta)
    
    ranked_candidates = []
    for i, cand in enumerate(candidates):
        # Core HTOM Score
        # Expected Improvement Score
        # If p_acc is low, the rest don't matter much.
        score = (w_acc * p_acc[i]) + \
                (w_cons * p_acc[i] * p_cons[i]) + \
                (w_speed * p_acc[i] * p_cons[i] * p_speed[i])
        
        # Difficulty Penalty (Heuristic guidance)
        diff = cand.get('difficulty', 5.0)
        dist = abs(diff - target_difficulty)
        penalty = 1.0 / (1.0 + 0.5 * dist) # Soft penalty
        
        final_score = score * penalty
        
        ranked_candidates.append({
            **cand,
            "score": float(final_score),
            "debug_info": {
                "p_acc": float(p_acc[i]),
                "p_cons": float(p_cons[i]),
                "p_speed": float(p_speed[i]),
                "raw_score": float(score),
                "diff_penalty": float(penalty)
            }
        })
        
    # Sort descending
    ranked_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return ranked_candidates
