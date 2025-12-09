import numpy as np
from typing import Dict, List, Optional

def update_long_term_features(
    current_state: Dict, 
    new_vector: List[float], 
    alpha: float = 0.1
) -> Dict:
    """
    Updates the long-term feature state with a new short-term vector using EMA.
    
    Args:
        current_state: Dictionary containing 'ema_mean', 'ema_var', and 'count'.
        new_vector: The new short-term feature vector (list of floats).
        alpha: The smoothing factor for EMA (default 0.1).
        
    Returns:
        Updated state dictionary.
    """
    new_vec = np.array(new_vector, dtype=np.float64)
    
    # Initialize if empty or malformed
    if not current_state or 'ema_mean' not in current_state:
        return {
            'ema_mean': new_vec.tolist(),
            'ema_var': np.zeros_like(new_vec).tolist(),
            'count': 1
        }
        
    # Parse current state
    # Ensure dimensions match; if new vector is different size, reset (or handle gracefully)
    # For now, assuming fixed dimension from UserFeatureExtractor
    ema_mean = np.array(current_state['ema_mean'], dtype=np.float64)
    if ema_mean.shape != new_vec.shape:
        # Fallback reset if dimension changed (e.g. code update)
        return {
            'ema_mean': new_vec.tolist(),
            'ema_var': np.zeros_like(new_vec).tolist(),
            'count': 1
        }

    ema_var = np.array(current_state.get('ema_var', np.zeros_like(ema_mean)), dtype=np.float64)
    count = current_state.get('count', 0)
    
    # EMA Update Logic
    # 1. Update Mean: mu_t = (1-a)*mu_{t-1} + a*x_t
    diff = new_vec - ema_mean
    new_mean = ema_mean + alpha * diff
    
    # 2. Update Variance: var_t = (1 - alpha) * (var_old + alpha * diff^2)
    # Derived from Welford's for EMA
    new_var = (1 - alpha) * (ema_var + alpha * (diff ** 2))

    return {
        'ema_mean': new_mean.tolist(),
        'ema_var': new_var.tolist(),
        'count': count + 1
    }

def get_long_term_vector(state: Dict) -> List[float]:
    """
    Returns the concatenated vector [EMA_Mean, EMA_StdDev] from the state.
    """
    if not state or 'ema_mean' not in state:
        return []
        
    ema_mean = np.array(state['ema_mean'], dtype=np.float64)
    ema_var = np.array(state['ema_var'], dtype=np.float64)
    
    # Clip variance to be non-negative (numerical stability)
    ema_std = np.sqrt(np.maximum(ema_var, 0))
    
    return np.concatenate([ema_mean, ema_std]).tolist()
