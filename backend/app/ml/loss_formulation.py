import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from app.ml.user_encoder import UserEncoder
from app.ml.snippet_tower import SnippetEncoder
from app.config import settings

class TwoTowerTrainingWrapper(nn.Module):
    """
    Wraps User and Snippet towers plus the hierarchical heads (Flow, Growth)
    for training purposes.
    """
    def __init__(self, user_encoder: UserEncoder, snippet_encoder: SnippetEncoder):
        super().__init__()
        self.user_encoder = user_encoder
        self.snippet_encoder = snippet_encoder
        self.embedding_dim = settings.embedding_dim

        # Shared embedding dimension is assumed to be output_dim of towers (e.g. 30)
        input_dim = self.embedding_dim * 2 # Concatenated u, s

        # --- Hierarchical Heads ---
        
        # Level 2: Flow Head (Predicts probability of Flow state [0,1])
        self.flow_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output logit (will use BCEWithLogitsLoss)
        )

        # Level 3: Growth Head (Predicts growth metric [-1, 1])
        self.growth_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1), 
            nn.Tanh() # Enforce range [-1, 1]
        )
        
        # Note: Ranking (Level 1) is computed via dot product of embeddings
        # Note: Difficulty (Level 4) is a constraint on metadata, no head needed

    def forward(
        self,
        # User Inputs
        user_short_term_seq: torch.Tensor, # (B, T, 12)
        user_session_stats: torch.Tensor,  # (B, 5)
        user_long_term: torch.Tensor,      # (B, 40)
        # Snippet Inputs
        snippet_features_pos: torch.Tensor, # (B, 30) - Positive sample
        snippet_features_neg: torch.Tensor  # (B, K, 30) - Negative samples
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Compute User Embedding
        # (B, D)
        u_emb = self.user_encoder(user_short_term_seq, user_session_stats, user_long_term)
        
        # 2. Compute Snippet Embeddings
        # Positive: (B, D)
        s_pos_emb = self.snippet_encoder(snippet_features_pos)
        
        # Negatives: (B, K, D)
        # Flatten to pass through encoder: (B*K, 30)
        B, K, _ = snippet_features_neg.shape
        s_neg_flat = snippet_features_neg.view(B * K, -1)
        s_neg_emb_flat = self.snippet_encoder(s_neg_flat)
        s_neg_emb = s_neg_emb_flat.view(B, K, -1)

        # 3. Level 1: Ranking Scores (Dot Product)
        # Positive Score: (B, 1)
        # (B, D) * (B, D) -> (B, 1)
        score_pos = (u_emb * s_pos_emb).sum(dim=1, keepdim=True)
        
        # Negative Scores: (B, K)
        # (B, 1, D) * (B, K, D) -> (B, K)
        score_neg = (u_emb.unsqueeze(1) * s_neg_emb).sum(dim=2)

        # 4. Hierarchical Heads Inputs
        # Concatenate User + Positive Snippet: (B, 2*D)
        combined_features = torch.cat([u_emb, s_pos_emb], dim=1)

        # 5. Level 2: Flow Prediction
        # (B, 1)
        flow_logit = self.flow_head(combined_features)

        # 6. Level 3: Growth Prediction
        # (B, 1)
        growth_pred = self.growth_head(combined_features)

        return {
            "u_emb": u_emb,
            "s_pos_emb": s_pos_emb,
            "score_pos": score_pos,       # (B, 1)
            "score_neg": score_neg,       # (B, K)
            "flow_logit": flow_logit,     # (B, 1)
            "growth_pred": growth_pred    # (B, 1)
        }

def compute_hierarchical_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    hyperparams: Dict[str, float]
) -> Dict[str, torch.Tensor]:
    """
    Computes the combined hierarchical loss.
    """
    
    # Unpack Outputs
    score_pos = outputs["score_pos"] # (B, 1)
    score_neg = outputs["score_neg"] # (B, K)
    flow_logit = outputs["flow_logit"] # (B, 1)
    growth_pred = outputs["growth_pred"] # (B, 1)
    
    # Unpack Targets
    y_flow = targets["y_flow"]      # (B, 1) in [0, 1]
    y_growth = targets["y_growth"]  # (B, 1) in [-1, 1]
    d_snippet = targets["d_snippet"] # (B, 1) normalized difficulty
    c_user = targets["c_user"]       # (B, 1) comfort boundary
    s_user = targets["s_user"]       # (B, 1) struggle boundary

    device = score_pos.device

    # --- Level 1: Ranking Loss (InfoNCE / Cross Entropy) ---
    # Concatenate logits: [Pos, Neg1, Neg2...] -> (B, 1+K)
    logits = torch.cat([score_pos, score_neg], dim=1)
    # Target is always index 0 (the positive sample)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    
    L_rank = F.cross_entropy(logits, labels)

    # --- Level 2: Flow Loss (BCE) ---
    L_flow = F.binary_cross_entropy_with_logits(flow_logit, y_flow)

    # --- Level 3: Growth Loss (Masked MSE) ---
    tau_flow = hyperparams.get("tau_flow", 0.3)
    
    # Create mask: Only compute growth loss where flow was decent
    # y_flow is ground truth [0,1]
    mask_growth = (y_flow > tau_flow).float()
    
    L_growth_raw = F.mse_loss(growth_pred, y_growth, reduction='none')
    L_growth = (mask_growth * L_growth_raw).sum() / (mask_growth.sum() + 1e-6)

    # --- Level 4: Difficulty Boost Loss (Masked Hinge) ---
    # Mask: Flow was decent AND Growth was non-negative (didn't regress)
    mask_diff = ((y_flow > tau_flow) & (y_growth >= 0)).float()
    
    margin_low = hyperparams.get("margin_low", 0.05)
    margin_high = hyperparams.get("margin_high", 0.0)
    
    # Penalize if too easy: d < c + margin
    # ReLU( (c + m) - d ) -> positive if d too small
    too_easy = torch.relu((c_user + margin_low) - d_snippet)
    
    # Penalize if too hard: d > s - margin
    # ReLU( d - (s - m) ) -> positive if d too big
    too_hard = torch.relu(d_snippet - (s_user - margin_high))
    
    L_diff_raw = too_easy + too_hard
    L_diff = (mask_diff * L_diff_raw).sum() / (mask_diff.sum() + 1e-6)

    # --- Total Loss ---
    lambda_flow = hyperparams.get("lambda_flow", 0.5)
    lambda_growth = hyperparams.get("lambda_growth", 0.3)
    lambda_diff = hyperparams.get("lambda_diff", 0.2)
    
    total_loss = L_rank + (lambda_flow * L_flow) + (lambda_growth * L_growth) + (lambda_diff * L_diff)

    return {
        "total": total_loss,
        "L_rank": L_rank,
        "L_flow": L_flow,
        "L_growth": L_growth,
        "L_diff": L_diff
    }