import os
from typing import Dict, Tuple, Optional, List, Any, Union

import numpy as np
import torch
import torch.nn as nn

from app.config import settings
from app.models.schema import UserState, KeystrokeEvent
from app.ml.user_features import UserFeatureExtractor

# ============================================================
# Global model
# ============================================================

_USER_ENCODER: Optional["UserEncoder"] = None

# ============================================================
# Model definition
# ============================================================

class UserEncoder(nn.Module):
    """
    User-tower encoder with Long-Term + Short-Term memory.
    
    Architecture:
      1. Long-Term Path:
         UserFeatureExtractor features (40-dim) -> MLP -> LongTermEmbedding (hidden_dim)
         
      2. Short-Term Path (Sequential):
         Sequence of Short-term vectors (T, 12) -> GRU(h0=LongTermEmbedding) -> ShortTermEmbedding (hidden_dim)
         
      3. Current Session Stats Path:
         Current session rolling stats (5-dim) -> MLP -> SessionStatsEmbedding (32-dim)

      4. Combination:
         Concat(LongTermEmbedding, ShortTermEmbedding, SessionStatsEmbedding) -> FinalMLP -> UserEmbedding (D-dim)
    """

    def __init__(
        self,
        short_term_dim: int = 12, # From UserFeatureExtractor.compute_short_term_features
        session_stats_dim: int = 5,
        long_term_dim: int = 40,  # UserFeatureExtractor.feature_dim()
        hidden_dim: int = 64, # Shared hidden dim for GRU and LongTerm projection
        output_dim: int = 30,
    ):
        super().__init__()
        
        # --- Long Term Feature Encoder ---
        # Projects 40-dim long-term features into hidden_dim (e.g., 64)
        # Used as the initial state (h0) for the GRU to provide context
        self.long_term_mlp = nn.Sequential(
            nn.Linear(long_term_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Output H-dim
            nn.Tanh() 
        )

        # --- Short Term Sequence Encoder (GRU) ---
        # Processes sequence of behavioral vectors
        self.gru = nn.GRU(
            input_size=short_term_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # --- Session Stats Encoder ---
        # Projects 5-dim rolling session stats into a 32-dim embedding
        self.session_stats_mlp = nn.Sequential(
            nn.Linear(session_stats_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32), # Output 32-dim
            nn.ReLU(),
        )

        # --- Final Combination Head ---
        # Inputs: LongTermEmbedding (H) + GRU_Final_State (H) + SessionStatsEmbedding (32)
        self.head = nn.Linear(hidden_dim + hidden_dim + 32, output_dim)

    def forward(
        self,
        short_term_sequence: torch.Tensor, # (B, SeqLen, 12)
        session_stats: torch.Tensor,        # (B, 5)
        long_term_stats: torch.Tensor,      # (B, 40)
    ) -> torch.Tensor:
        """
        Returns:
            embedding: (B, output_dim)
        """
        B = session_stats.shape[0]

        # 1. Compute Long Term Embedding (B, H)
        long_term_emb = self.long_term_mlp(long_term_stats) 

        # 2. Short-Term Sequence Encoding (GRU)
        # Use Long-Term embedding as initial hidden state h0
        # h0 shape: (num_layers=1, B, H)
        h0 = long_term_emb.unsqueeze(0)

        if short_term_sequence is not None and short_term_sequence.shape[1] > 0:
            # out: (B, T, H), h_n: (1, B, H)
            _, h_n = self.gru(short_term_sequence, h0)
            short_term_emb = h_n[-1] # (B, H)
        else:
            # Fallback if no sequence provided, use the initial state (long-term context)
            short_term_emb = h0.squeeze(0)

        # 3. Compute Session Stats Embedding (B, 32)
        sess_feat_emb = self.session_stats_mlp(session_stats)

        # 4. Combine All Contexts
        combined = torch.cat([long_term_emb, short_term_emb, sess_feat_emb], dim=-1) # (B, H + H + 32)
        
        embedding = self.head(combined) # (B, D)

        return embedding


# ============================================================
# Model getter
# ============================================================

def get_model() -> UserEncoder:
    global _USER_ENCODER
    if _USER_ENCODER is None:
        _USER_ENCODER = UserEncoder(output_dim=settings.embedding_dim)
        _USER_ENCODER.eval()
    return _USER_ENCODER


# ============================================================
# Helper: Tensor Builders
# ============================================================

def _build_session_stats_tensor(state: UserState) -> torch.Tensor:
    """
    Normalize rolling stats into a (1, 5) tensor.
    """
    wpm_norm = min(state.rollingWpm / 200.0, 1.0)
    acc_norm = float(state.rollingAccuracy)
    back_norm = min(state.backspaceRate / 0.2, 1.0)
    diff_norm = state.currentDifficulty / 10.0
    hes_norm = min(state.hesitationCount / 10.0, 1.0)

    return torch.tensor(
        [[wpm_norm, acc_norm, back_norm, diff_norm, hes_norm]],
        dtype=torch.float32,
    )

def _build_long_term_features_tensor(
    user_features_dict: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Rehydrates UserFeatureExtractor and computes the 40-dim long-term feature vector.
    """
    extractor = UserFeatureExtractor.from_dict(user_features_dict or {})
    vec = extractor.compute_user_features()
    return torch.tensor([vec], dtype=torch.float32)


# ============================================================
# Main Public API
# ============================================================

def get_user_embedding(
    state: UserState,
    user_features_dict: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Computes the user embedding vector.
    
    Args:
        state: Current session state (rolling stats, recent keystrokes).
        user_features_dict: Optional dictionary of long-term feature state (from DB JSON).
    """
    model = get_model()

    # 1. Session Stats Tensor (B, 5)
    session_tensor = _build_session_stats_tensor(state)

    # 2. Long-Term Tensor (B, 40)
    long_term_tensor = _build_long_term_features_tensor(user_features_dict)
    
    # 3. Build Short-Term Sequence (B, T, 12)
    extractor = UserFeatureExtractor.from_dict(user_features_dict or {})
    
    # A. Retrieve history (last 10 items)
    history_list = extractor.short_term_history[-10:] # List[List[float]]
    
    # B. Compute CURRENT session vector (if valid keystrokes exist)
    current_vec = None
    if state.recentKeystrokes and len(state.recentKeystrokes) > 0:
        session_data = {
            'keystroke_events': [e.dict() for e in state.recentKeystrokes],
            'wpm': state.rollingWpm,
            'accuracy': state.rollingAccuracy,
            'completed': True, 
            'quit_progress': 1.0
        }
        current_vec = extractor.compute_short_term_features(session_data).tolist()
    
    # C. Construct Sequence
    # Sequence = [Historical_1, Historical_2, ..., Current]
    full_sequence = [np.array(v, dtype=np.float32) for v in history_list]
    if current_vec:
        full_sequence.append(np.array(current_vec, dtype=np.float32))
        
    if not full_sequence:
        # If completely empty (cold start + no current keystrokes), pass empty tensor
        # GRU will handle T=0 case by falling back to h0 (long-term context)
        short_term_tensor = torch.zeros(1, 0, 12, dtype=torch.float32)
    else:
        # Stack into (1, T, 12)
        short_term_tensor = torch.tensor(np.array([full_sequence]), dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        embedding = model(short_term_tensor, session_tensor, long_term_tensor)

    return embedding.numpy()[0]