import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class UserEncoder(nn.Module):
    """
    Encodes user features (global stats + pooled history) into a latent vector.
    Input: ~60-80 dimensions (depending on exact feature count from UserFeatureExtractor)
    Output: 32 dimensions
    """
    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SnippetEncoder(nn.Module):
    """
    Encodes snippet manual features into a latent vector.
    Input: ~15 dimensions
    Output: 32 dimensions
    """
    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # No LayerNorm usually needed for small snippet feats, but consistent with User
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HTOM(nn.Module):
    """
    Hierarchical Typing Outcome Model.
    Combines User and Snippet vectors to predict:
    1. Accuracy (Success)
    2. Consistency (given Accuracy)
    3. Speed (given Accuracy + Consistency)
    """
    def __init__(self, user_input_dim: int, snippet_input_dim: int):
        super().__init__()
        self.user_encoder = UserEncoder(user_input_dim)
        self.snippet_encoder = SnippetEncoder(snippet_input_dim)
        
        # Combined latent space size
        combined_dim = 32 + 32 
        
        # Core hidden layer
        self.hidden = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU()
        )
        
        # Heads
        # 1. Accuracy Head
        self.acc_head = nn.Linear(64, 1)
        
        # 2. Consistency Head (Conditioned on Acc prob)
        self.cons_head = nn.Linear(64 + 1, 1)
        
        # 3. Speed Head (Conditioned on Acc + Cons probs)
        self.speed_head = nn.Linear(64 + 1 + 1, 1)

    def forward(self, user_feats: torch.Tensor, snippet_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        u_vec = self.user_encoder(user_feats)
        s_vec = self.snippet_encoder(snippet_feats)
        
        # Combine
        combined = torch.cat([u_vec, s_vec], dim=-1)
        h = self.hidden(combined)
        
        # Accuracy
        p_acc_logit = self.acc_head(h)
        p_acc = torch.sigmoid(p_acc_logit)
        
        # Consistency (concat p_acc)
        h_cons = torch.cat([h, p_acc], dim=-1)
        p_cons_logit = self.cons_head(h_cons)
        p_cons = torch.sigmoid(p_cons_logit)
        
        # Speed (concat p_acc, p_cons)
        h_speed = torch.cat([h, p_acc, p_cons], dim=-1)
        p_speed_logit = self.speed_head(h_speed)
        p_speed = torch.sigmoid(p_speed_logit)
        
        return {
            "p_acc": p_acc,
            "p_cons": p_cons,
            "p_speed": p_speed
        }

# Global instance
_HTOM_MODEL = None

def get_htom_model(user_dim: int = 57, snippet_dim: int = 15) -> HTOM:
    global _HTOM_MODEL
    if _HTOM_MODEL is None:
        _HTOM_MODEL = HTOM(user_dim, snippet_dim)
        _HTOM_MODEL.eval()
    return _HTOM_MODEL
