import numpy as np
import torch
from typing import List, Dict, Any
from app.config import settings
from app.ml.user_encoder import get_user_embedding
from torch import nn

# global lazy-loaded model
_RANKER_MODEL = None

class TwoTowerRanker(nn.Module):
    """
    Scores a (User, Snippet) pair.
    """
    def __init__(self, embedding_dim: int = 30):
        super().__init__()
        # A simple bilinear layer to allow for interaction weighting
        # score = x1 * W * x2 + b
        self.bilinear = nn.Bilinear(embedding_dim, embedding_dim, 1)
        
    def forward(self, user_emb: torch.Tensor, snippet_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_emb: (Batch, Dim)
            snippet_emb: (Batch, Dim)
        Returns:
            score: (Batch, 1)
        """
        return self.bilinear(user_emb, snippet_emb)

# ---------------------------------------------------------
# Load / init the Two-Tower model
# ---------------------------------------------------------
def get_ranker_model():
    global _RANKER_MODEL

    if _RANKER_MODEL is None:
        _RANKER_MODEL = TwoTowerRanker(embedding_dim=settings.embedding_dim)
        _RANKER_MODEL.eval()

        # TODO: load trained weights
        # state = torch.load(settings.ranker_model_path, map_location="cpu")
        # _RANKER_MODEL.load_state_dict(state)

    return _RANKER_MODEL


# ---------------------------------------------------------
# Ranking Function (new pipeline)
# ---------------------------------------------------------
def rank_snippets(
    user_embedding: np.ndarray,
    candidates: List[dict],
    target_difficulty: float,
    exploration_bonus: float = 0.0,
) -> List[dict]:
    """
    Ranks candidate snippets using the Two-Tower model if snippet embeddings exist.
    Otherwise falls back to difficulty + heuristic score.

    Inputs:
      user_embedding    – pre-computed user vector (numpy array)
      candidates        – list of snippet dicts from DB/FAISS
      target_difficulty – desired difficulty level
      exploration_bonus – optional RL exploration weight

    Returns:
      ranked list of candidate dicts with score + debug_info
    """

    if not candidates:
        return []

    # -----------------------------------------------------
    # Step 1: Prepare User Tensor
    # -----------------------------------------------------
    user_tensor = torch.tensor(user_embedding, dtype=torch.float32).unsqueeze(0)

    # -----------------------------------------------------
    # Step 2: Check if snippet embeddings exist
    # -----------------------------------------------------
    has_embeddings = all("embedding" in c and c["embedding"] is not None for c in candidates)

    ranker = get_ranker_model()
    ranked = []

    # =====================================================
    # Full Two-Tower scoring (preferred)
    # =====================================================
    if has_embeddings:
        snippet_matrix = np.stack([np.array(c["embedding"], dtype=np.float32) for c in candidates])
        snippet_tensor = torch.tensor(snippet_matrix, dtype=torch.float32)

        # Broadcast user → batch
        user_batch = user_tensor.expand(len(candidates), -1)

        with torch.no_grad():
            # shape: (N, 1)
            scores = ranker(user_batch, snippet_tensor).squeeze(1).numpy()

        for score, cand in zip(scores, candidates):
            final_score = float(score) + exploration_bonus

            ranked.append({
                **cand,
                "score": final_score,
                "debug_info": {
                    "model_score": float(score),
                    "exploration_bonus": exploration_bonus,
                    "mode": "two_tower"
                }
            })

    else:
        # Fallback: Heuristic ranking based on calibrated difficulty
        # This path is taken if embeddings are missing (e.g., during init/migration)
        # We use a "Zone of Proximal Development" strategy: 
        # Rank snippets closest to the user's current difficulty level.
        
        # Extract target difficulty (default to 5.0 if missing)
        # In the new signature, we don't have user_state dict, but we have target_difficulty passed in.
        
        for cand in candidates:
            # difficulty_score is the calibrated 1-10 value
            diff = cand.get("difficulty")
            
            if diff is None:
                print(f"[WARNING] Snippet {cand.get('id')} has None difficulty. Defaulting to 5.0")
                diff = 5.0
                
            dist = abs(diff - target_difficulty)
            
            # Score: Higher is better. Max score 1.0 when dist is 0.
            score = 1.0 / (1.0 + dist)
            
            ranked.append({
                **cand,
                "score": score,
                "debug_info": {
                    "mode": "difficulty_heuristic",
                    "diff_dist": dist
                }
            })

    # -----------------------------------------------------
    # Step 3: Sort (descending)
    # -----------------------------------------------------
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

if __name__ == "__main__":
    from app.models.schema import UserState, KeystrokeEvent

    # Create a valid test UserState instance
    user_state_obj = UserState(
        rollingWpm=60.0,
        rollingAccuracy=0.92,
        backspaceRate=0.08,
        hesitationCount=3,
        recentErrors=["t", "h"],
        currentDifficulty=4.7,
        recentSnippetIds=[],

        recentKeystrokes=[
            KeystrokeEvent(timestamp=10, key="a", isBackspace=False, isCorrect=True),
            KeystrokeEvent(timestamp=25, key="b", isBackspace=True,  isCorrect=False),
        ],

    )

    # Fake snippets
    candidates = [
        {
            "id": "A",
            "text": "example snippet here",
            "difficulty_score": 4.2,
            "embedding": np.random.randn(30).astype("float32").tolist(),
        },
        {
            "id": "B",
            "text": "another practice line",
            "difficulty_score": 5.1,
            "embedding": np.random.randn(30).astype("float32").tolist(),
        },
    ]

    ranked = rank_snippets(
        user_state=user_state_obj, # type: ignore
        candidates=candidates,
        target_difficulty=4.5,
        exploration_bonus=0.1,
    )

    print("\n=== Test Ranking Output ===")
    for r in ranked:
        print(r["id"], r["score"], r["debug_info"])
