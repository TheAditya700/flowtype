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

def expected_abs_gaussian(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    E[|X|] for X ~ N(mu, sigma^2), elementwise.
    """
    sigma = np.clip(sigma, 1e-8, None)
    term1 = sigma * np.sqrt(2.0 / np.pi) * np.exp(-(mu ** 2) / (2.0 * sigma ** 2))
    term2 = np.abs(mu) * (1.0 - 2.0 * norm.cdf(-np.abs(mu) / sigma))
    return term1 + term2


def prob_abs_gt_eps(mu: np.ndarray, sigma: np.ndarray, eps: float) -> np.ndarray:
    """
    P(|X| > eps) for X ~ N(mu, sigma^2), elementwise.
    """
    sigma = np.clip(sigma, 1e-8, None)
    p_pos = 1.0 - norm.cdf(eps, loc=mu, scale=sigma)
    p_neg = norm.cdf(-eps, loc=mu, scale=sigma)
    return p_pos + p_neg


def compute_top_user_components(
    agent,
    k: int = 5,
    eps: float = 1e-3,
    exclude_prev_snippet: bool = True,
):
    """
    Rank user features by:
      impact_j      = sum_i E[|W_ij|]
            certainty_j   = mean_i P(|W_ij| > eps)
            uncertainty_j = 1 - certainty_j  (remove impact component)
    """
    W_mean = agent.W_mean              # (16, 130)
    W_prec = agent.W_precision
    W_var = 1.0 / W_prec
    W_std = np.sqrt(W_var)

    E_abs = expected_abs_gaussian(W_mean, W_std)          # (16, 130)
    P_nonzero = prob_abs_gt_eps(W_mean, W_std, eps)       # (16, 130)

    impact = E_abs.sum(axis=0)                            # (130,)
    certainty = P_nonzero.mean(axis=0)                    # (130,)
    uncertainty = 1.0 - certainty                         # (130,)

    valid_mask = np.ones(W_mean.shape[1], dtype=bool)
    if exclude_prev_snippet:
        valid_mask[114:130] = False

    def topk(score_vec, k_):
        masked = np.where(valid_mask, score_vec, -np.inf)
        return np.argsort(masked)[-k_:][::-1]

    top_impact_idx = topk(impact, k)
    top_certain_idx = topk(certainty, k)
    top_uncertain_idx = topk(uncertainty, k)

    def feature_info(j):
        return {
            "user_feature_idx": int(j),
            "impact": float(impact[j]),
            "certainty": float(certainty[j]),
            "uncertainty": float(uncertainty[j]),
            "mean_weight": float(np.mean(W_mean[:, j])),
            "mean_precision": float(np.mean(W_prec[:, j])),
        }

    return {
        "top_impact": [feature_info(j) for j in top_impact_idx],
        "top_certain": [feature_info(j) for j in top_certain_idx],
        "top_uncertain": [feature_info(j) for j in top_uncertain_idx],
    }


def compute_top_certain_uncertain(agent, k=10):
    """
    Backward-compatible wrapper returning (top_certain, top_uncertain, top_impact).
    """
    res = compute_top_user_components(agent, k=k, eps=1e-3, exclude_prev_snippet=True)
    return (res["top_certain"], res["top_uncertain"], res["top_impact"])
