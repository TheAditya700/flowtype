import numpy as np
from scipy.stats import norm

def compute_model_snapshot(agent, prev_snapshot=None):
    W_mean = agent.W_mean
    W_prec = agent.W_precision
    W_var = 1.0 / W_prec

    abs_mean = np.abs(W_mean)

    # --- Belief confidence ---
    mean_precision = float(np.mean(W_prec))
    median_precision = float(np.median(W_prec))
    p90_precision = float(np.percentile(W_prec, 90))
    p99_precision = float(np.percentile(W_prec, 99))
    mean_variance = float(np.mean(W_var))

    fraction_high_confidence = float(np.mean(W_prec > 100.0))

    # --- Belief structure ---
    mean_abs_weight = float(np.mean(abs_mean))
    p90_abs_weight = float(np.percentile(abs_mean, 90))

    fraction_near_zero_mean = float(np.mean(abs_mean < 1e-3))
    fraction_confident_irrelevant = float(
        np.mean((abs_mean < 1e-3) & (W_prec > 100.0))
    )

    # --- Learning dynamics (vs previous snapshot) ---
    if prev_snapshot is None:
        mean_abs_delta_mean = 0.0
        mean_delta_precision = 0.0
        fraction_weights_updated = 0.0
    else:
        delta_mean = np.abs(W_mean - prev_snapshot["W_mean"])
        delta_prec = W_prec - prev_snapshot["W_precision"]

        mean_abs_delta_mean = float(np.mean(delta_mean))
        mean_delta_precision = float(np.mean(delta_prec))
        fraction_weights_updated = float(np.mean(delta_prec > 1e-6))

    return {
        "mean_precision": mean_precision,
        "median_precision": median_precision,
        "p90_precision": p90_precision,
        "p99_precision": p99_precision,
        "mean_variance": mean_variance,
        "fraction_high_confidence": fraction_high_confidence,

        "mean_abs_weight": mean_abs_weight,
        "p90_abs_weight": p90_abs_weight,
        "fraction_near_zero_mean": fraction_near_zero_mean,
        "fraction_confident_irrelevant": fraction_confident_irrelevant,

        "mean_abs_delta_mean": mean_abs_delta_mean,
        "mean_delta_precision": mean_delta_precision,
        "fraction_weights_updated": fraction_weights_updated,
    }

def compute_top_interactions(agent, k=10):
    W_mean = agent.W_mean
    W_prec = agent.W_precision
    W_var = 1.0 / W_prec

    mu = W_mean
    sigma = np.sqrt(W_var)

    # Probability weight is positive
    p_pos = 1.0 - norm.cdf(0.0, loc=mu, scale=sigma)
    p_neg = norm.cdf(0.0, loc=mu, scale=sigma)

    score_pos = mu * p_pos
    score_neg = -mu * p_neg

    flat_pos = score_pos.flatten()
    flat_neg = score_neg.flatten()

    top_pos_idx = np.argsort(flat_pos)[-k:][::-1]
    top_neg_idx = np.argsort(flat_neg)[-k:][::-1]

    def unpack(idx):
        i = idx // mu.shape[1]
        j = idx % mu.shape[1]
        return {
            "snippet_feature_idx": int(i),
            "user_feature_idx": int(j),
            "mean": float(mu[i, j]),
            "precision": float(W_prec[i, j]),
            "variance": float(W_var[i, j]),
            "p_positive": float(p_pos[i, j]),
        }

    return (
        [unpack(i) for i in top_pos_idx],
        [unpack(i) for i in top_neg_idx],
    )

def compute_top_certain_uncertain(agent, k=10):
    """
    Compute top K features by:
    - importance: actual contribution to predictions (weighted by typical magnitudes)
    - certain: most confident features (high precision, high contribution)
    - uncertain: least confident features (low precision, high variance)
    
    Excludes Previous Snippet PCA components (114-129).
    
    Returns:
        (top_certain, top_uncertain, top_importance): Three lists of feature information
    """
    W_mean = agent.W_mean  # Shape: (16 snippet features, 130 user features)
    W_prec = agent.W_precision
    W_var = 1.0 / W_prec
    
    # Calculate actual contribution: sum of |weight| across all snippet dimensions
    # This tells us how much each user feature actually impacts predictions
    user_contribution = np.sum(np.abs(W_mean), axis=0)  # Shape: (130,)
    
    # Filter out Prev Snippet PCA components (114-129)
    valid_mask = np.ones(130, dtype=bool)
    valid_mask[114:130] = False
    
    # Get indices sorted by contribution (for importance mode)
    valid_contribution = np.where(valid_mask, user_contribution, -np.inf)
    top_importance_indices = np.argsort(valid_contribution)[-k:][::-1]
    
    # For certain: high precision AND high contribution (weighted score)
    # Normalize precision to 0-1 scale (using log to handle wide range)
    norm_precision = np.log10(np.clip(W_prec, 1.0, 1e6))  # 0 to 6 range
    norm_precision = norm_precision / 6.0  # 0 to 1
    
    # Average certainty across snippet dimensions
    avg_precision = np.mean(norm_precision, axis=0)  # Shape: (130,)
    
    # Certainty score: precision * contribution (features that are certain AND matter)
    certainty_score = avg_precision * user_contribution
    valid_certainty = np.where(valid_mask, certainty_score, -np.inf)
    top_certain_indices = np.argsort(valid_certainty)[-k:][::-1]
    
    # For uncertain: low precision (high variance) but still has some contribution
    # We want features the model is learning about, not just noise
    avg_variance = np.mean(W_var, axis=0)  # Shape: (130,)
    
    # Uncertainty score: variance * contribution (uncertain features that matter)
    uncertainty_score = avg_variance * user_contribution
    valid_uncertainty = np.where(valid_mask, uncertainty_score, -np.inf)
    top_uncertain_indices = np.argsort(valid_uncertainty)[-k:][::-1]
    
    def create_feature_info(user_idx):
        """Create aggregated feature info for a user feature across all snippet dimensions."""
        return {
            "user_feature_idx": int(user_idx),
            "importance": float(user_contribution[user_idx]),
            "precision": float(np.mean(W_prec[:, user_idx])),
            "variance": float(np.mean(W_var[:, user_idx])),
            "mean_weight": float(np.mean(W_mean[:, user_idx])),
            "sign": "positive" if np.sum(W_mean[:, user_idx]) > 0 else "negative",
        }
    
    top_importance = [create_feature_info(idx) for idx in top_importance_indices]
    top_certain = [create_feature_info(idx) for idx in top_certain_indices]
    top_uncertain = [create_feature_info(idx) for idx in top_uncertain_indices]
    
    return (top_certain, top_uncertain, top_importance)
    
    return (
        [unpack_weight(i) for i in top_certain_idx],
        [unpack_weight(i) for i in top_uncertain_idx],
    )
