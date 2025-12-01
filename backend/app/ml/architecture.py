import torch
import torch.nn as nn
import torch.nn.functional as F

class UserEncoder(nn.Module):
    """
    Encodes user state and recent keystroke history into a latent embedding.
    
    Architecture:
    - GRU: Processes sequence of keystroke features (IKI, is_error, is_backspace).
    - MLP: Processes scalar user stats (WPM, Accuracy, etc.).
    - Fusion: Concatenates GRU state + MLP output -> Final Projection.
    """
    def __init__(
        self, 
        keystroke_input_dim: int = 3, 
        stats_input_dim: int = 5, 
        hidden_dim: int = 64, 
        output_dim: int = 30
    ):
        super().__init__()
        
        # 1. Keystroke Sequence Encoder (GRU)
        # Input: (Batch, Seq_Len, 3) -> Output: (Batch, Hidden)
        self.gru = nn.GRU(
            input_size=keystroke_input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 2. User Stats Encoder (MLP)
        # Input: (Batch, 5) -> Output: (Batch, Hidden)
        self.stats_mlp = nn.Sequential(
            nn.Linear(stats_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 3. Fusion Layer
        # Concatenate GRU hidden state + Stats encoding -> Output Dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, keystrokes: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keystrokes: Tensor of shape (Batch, Seq_Len, keystroke_input_dim)
            stats: Tensor of shape (Batch, stats_input_dim)
        Returns:
            embedding: Tensor of shape (Batch, output_dim)
        """
        # GRU Forward
        # We only care about the final hidden state of the last layer
        # out: (Batch, Seq, Hidden), hn: (Num_Layers, Batch, Hidden)
        if keystrokes.size(1) > 0:
            _, hn = self.gru(keystrokes) 
            gru_embedding = hn.squeeze(0) # (Batch, Hidden)
        else:
            # Handle empty sequence case with zero embedding
            gru_embedding = torch.zeros(keystrokes.size(0), self.gru.hidden_size, device=keystrokes.device)

        
        # Stats Forward
        stats_embedding = self.stats_mlp(stats) # (Batch, Hidden)
        
        # Fusion
        combined = torch.cat([gru_embedding, stats_embedding], dim=1)
        output = self.fusion(combined)
        
        # Normalize output vector to unit length (cosine similarity compatible)
        output = F.normalize(output, p=2, dim=1)
        
        return output


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
