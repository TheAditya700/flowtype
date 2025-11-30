import numpy as np
from typing import List


def rank_snippets(
    user_embedding: np.ndarray, 
    candidates: List[dict], 
    target_difficulty: float
) -> List[dict]:
    """
    Ranks candidate snippets based on a flow-based function.
    Balances semantic match (vector distance) and difficulty suitability.
    """
    if not candidates:
        return []

    # Weights for the scoring function
    w_semantic = 1.0
    w_diff = 0.5

    ranked_candidates = []
    
    for cand in candidates:
        distance = cand.get('distance', 1.0)
        difficulty = cand.get('difficulty', 5.0)
        
        # Semantic Score (Inverse of L2 distance)
        semantic_score = 1.0 / (1.0 + distance)
        
        # Difficulty Penalty
        diff_penalty = abs(difficulty - target_difficulty)
        
        # Final Combined Score (Higher is better)
        final_score = (semantic_score * w_semantic) - (diff_penalty * w_diff)
        
        ranked_candidates.append({
            **cand,
            'score': final_score,
            'debug_info': {
                'semantic_score': semantic_score,
                'diff_penalty': diff_penalty
            }
        })
    
    # Sort by final score descending (best match first)
    ranked_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return ranked_candidates

