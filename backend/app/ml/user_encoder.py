import numpy as np
import torch
import os
from typing import List
from app.models.schema import UserState, KeystrokeEvent
from app.config import settings
from app.ml.architecture import UserEncoder

# Global model instance
_USER_ENCODER = None

def get_model():
    global _USER_ENCODER
    if _USER_ENCODER is None:
        # Initialize model
        _USER_ENCODER = UserEncoder(output_dim=settings.embedding_dim)
        _USER_ENCODER.eval() # Inference mode
        
        # TODO: Load trained weights if they exist
        # model_path = "app/ml/models/user_encoder.pth"
        # if os.path.exists(model_path):
        #     _USER_ENCODER.load_state_dict(torch.load(model_path))
            
    return _USER_ENCODER

def process_keystrokes(events: List[KeystrokeEvent]) -> torch.Tensor:
    """
    Converts list of KeystrokeEvents to (Seq_Len, 3) tensor.
    Features: [Normalized IKI, is_backspace, is_error]
    """
    if not events or len(events) < 2:
        return torch.zeros(1, 0, 3) # Empty sequence
    
    # Sort by timestamp just in case
    sorted_events = sorted(events, key=lambda x: x.timestamp)
    
    processed = []
    for i in range(1, len(sorted_events)):
        prev = sorted_events[i-1]
        curr = sorted_events[i]
        
        # Calculate Inter-Key Interval (ms) -> normalize (cap at 1000ms)
        iki = (curr.timestamp - prev.timestamp)
        iki_norm = min(max(iki, 0), 1000) / 1000.0
        
        is_backspace = 1.0 if curr.isBackspace else 0.0
        is_error = 1.0 if not curr.isCorrect else 0.0
        
        processed.append([iki_norm, is_backspace, is_error])
        
    # Convert to tensor: (1, Seq_Len, 3) - Batch size 1
    return torch.tensor([processed], dtype=torch.float32)

def get_user_embedding(state: UserState) -> np.ndarray:
    """
    Encodes user state into a vector embedding using the UserEncoder model.
    """
    model = get_model()
    
    # 1. Process Stats
    # Normalize features
    wpm_norm = min(state.rollingWpm / 200.0, 1.0)
    acc_norm = state.rollingAccuracy
    back_norm = min(state.backspaceRate / 0.2, 1.0) # Cap at 20%
    diff_norm = state.currentDifficulty / 10.0
    hes_norm = min(state.hesitationCount / 10.0, 1.0) # Cap at 10 hesitations
    
    stats_tensor = torch.tensor([[
        wpm_norm, 
        acc_norm, 
        back_norm, 
        diff_norm, 
        hes_norm
    ]], dtype=torch.float32) # (1, 5)
    
    # 2. Process Keystrokes
    keystrokes_tensor = process_keystrokes(state.recentKeystrokes)
    
    # 3. Forward Pass
    with torch.no_grad():
        embedding = model(keystrokes_tensor, stats_tensor)
    
    # Return as numpy array (30,)
    return embedding.numpy()[0]