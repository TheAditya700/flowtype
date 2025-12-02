import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from app.config import settings

_SNIPPET_ENCODER = None

class SnippetEncoder(nn.Module):
    """
    Snippet Tower: Projects raw/normalized difficulty features (30-dim)
    into the shared user-snippet embedding space (30-dim).
    """
    def __init__(
        self, 
        input_dim: int = 30, 
        hidden_dim: int = 64, 
        output_dim: int = 30
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (Batch, 30)
        Returns:
            embeddings: (Batch, 30)
        """
        return self.mlp(features)

def get_snippet_model() -> SnippetEncoder:
    global _SNIPPET_ENCODER
    if _SNIPPET_ENCODER is None:
        _SNIPPET_ENCODER = SnippetEncoder(
            input_dim=30, # Fixed feature size from difficulty_features
            output_dim=settings.embedding_dim
        )
        _SNIPPET_ENCODER.eval()
    return _SNIPPET_ENCODER
