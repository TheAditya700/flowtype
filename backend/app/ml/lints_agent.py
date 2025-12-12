import numpy as np
import pickle
import os
from typing import Dict, List, Optional

MODEL_PATH = "app/ml/lints_model.pkl"

# ------------------------------------------------------------------
# User feature indexing (within the 57-dim EMA vector)
# ------------------------------------------------------------------
IDX_ACCURACY = 0              # EMA of accuracy
IDX_IKI_CV = 10               # Global IKI CV (Coefficient of Variation for Inter-Keystroke-Intervals)
IDX_WPM_EFFECTIVE = 21        # EMA of effective WPM (wpm * accuracy)
IDX_SPIKE_RATE = 27           # EMA of spike_rate (0..1)

USER_BASE_DIM = 57            # base user feature dimensionality
SNIPPET_DIM = 16              # PCA snippet embedding dim

# User state passed to the model is:
#   [EMA(57) | STD(57) | PrevSnippet(16)] = 130
USER_DIM = USER_BASE_DIM * 2 + SNIPPET_DIM  # 57*2 + 16 = 130


class LinTSAgent:
    """
    Diagonal Linear Thompson Sampling Agent for a bilinear reward model.

    Model: R ≈ v^T W u
      - v: snippet embedding (16-dim PCA)
      - u: user state (130-dim: EMA(57) + Std(57) + prev_snippet(16))
      - W: (16 x 130) weight matrix

    We maintain a diagonal Gaussian posterior over W:
      W_ij ~ N(mean_ij, 1 / precision_ij)

    Prediction:
      - Sample W from the posterior
      - Compute query_vector q = W_sample @ user_state   (16-dim)
      - Use q to query FAISS / nearest snippets.

    Update:
      - Given (u, v, reward), update each W_ij using Bayesian diagonal regression.
    """

    def __init__(
        self,
        lambda_prior: float = 1.0,
        min_var: float = 1e-4,
        max_precision: float = 1e6,
        mean_lr: float = 1.0,
    ):
        """
        Args:
            lambda_prior: prior precision for all weights (L2 strength).
            min_var: minimum variance for Thompson sampling to avoid overconfidence.
            max_precision: upper cap on precision to avoid numerical blowup.
            mean_lr: "learning rate" for mean updates (1.0 = pure Bayesian update).
        """
        self.lambda_prior = lambda_prior
        self.min_var = float(min_var)
        self.max_precision = float(max_precision)
        self.mean_lr = float(mean_lr)

        # Mean matrix (16 x 130)
        self.W_mean = np.zeros((SNIPPET_DIM, USER_DIM), dtype=np.float32)

        # Diagonal precision for each weight
        self.W_precision = np.full(
            (SNIPPET_DIM, USER_DIM),
            self.lambda_prior,
            dtype=np.float32
        )

        self.version = 2  # incremented version

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------
    def get_context(
        self,
        user_ema: List[float],
        user_std: List[float],
        prev_snippet_embedding: Optional[List[float]],
    ) -> np.ndarray:
        """
        Constructs the 130-dim user state vector:

            u = [EMA(57) | STD(57) | PrevSnippetEmbedding(16)]

        Assumptions:
            - user_ema is 57-dim EMA over user features.
            - user_std is 57-dim stddev over user features.
            - prev_snippet_embedding is 16-dim PCA snippet vector, or None.
        """
        if prev_snippet_embedding is None:
            prev_snippet_embedding = [0.0] * SNIPPET_DIM

        u_ema = np.array(user_ema, dtype=np.float32)
        u_std = np.array(user_std, dtype=np.float32)
        u_prev = np.array(prev_snippet_embedding, dtype=np.float32)

        # Safety resize if slightly off
        if u_ema.shape[0] != USER_BASE_DIM:
            u_ema = np.resize(u_ema, USER_BASE_DIM)
        if u_std.shape[0] != USER_BASE_DIM:
            u_std = np.resize(u_std, USER_BASE_DIM)
        if u_prev.shape[0] != SNIPPET_DIM:
            u_prev = np.resize(u_prev, SNIPPET_DIM)

        user_state = np.concatenate([u_ema, u_std, u_prev], axis=0)
        assert user_state.shape[0] == USER_DIM
        return user_state

    # ------------------------------------------------------------------
    # Thompson Sampling prediction
    # ------------------------------------------------------------------
    def predict(self, user_state: np.ndarray) -> List[float]:
        """
        Samples W ~ N(mean, var) and returns:

            query_vector q = W_sample @ user_state   (16-dim)

        This query vector is the "ideal" snippet embedding: you pass it
        to FAISS (or similar) to find nearest snippets.
        """
        if user_state.shape[0] != USER_DIM:
            raise ValueError(
                f"user_state dim mismatch: expected {USER_DIM}, got {user_state.shape[0]}"
            )

        # Variance = 1 / precision, but keep a minimum variance for exploration
        var = 1.0 / self.W_precision + self.min_var
        std_devs = np.sqrt(var).astype(np.float32)

        # Thompson sample
        W_sample = np.random.normal(self.W_mean, std_devs)

        # q (16) = W (16x130) @ u (130)
        query_vector = W_sample @ user_state  # shape: (16,)

        return query_vector.astype(np.float32).tolist()

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------
    def update(self, user_state: np.ndarray, snippet_vector: np.ndarray, reward: float):
        """
        Update the posterior over W given one interaction:

            u: 130-dim user state
            v: 16-dim snippet embedding
            reward: scalar hierarchical reward

        Model: reward ≈ sum_ij W_ij * v_i * u_j
        Diagonal approximation: each W_ij is updated independently.
        """
        u = user_state.astype(np.float32)
        v = np.array(snippet_vector, dtype=np.float32)

        if u.shape[0] != USER_DIM:
            raise ValueError(
                f"user_state dim mismatch in update: expected {USER_DIM}, got {u.shape[0]}"
            )
        if v.shape[0] != SNIPPET_DIM:
            raise ValueError(
                f"snippet_vector dim mismatch: expected {SNIPPET_DIM}, got {v.shape[0]}"
            )

        # Outer product: X_ij = v_i * u_j
        X = np.outer(v, u)  # shape: (16, 130)

        # Old precision & mean
        P_old = self.W_precision
        M_old = self.W_mean

        # Update precision: P_new = P_old + X^2
        P_new = P_old + (X ** 2)

        # Clip precision to avoid numerical blowup
        np.clip(P_new, self.lambda_prior, self.max_precision, out=P_new)

        # Bayesian mean update:
        # M_bayes = (P_old * M_old + X * reward) / P_new
        M_bayes = (P_old * M_old + X * reward) / P_new

        # Optional "learning-rate" smoothing toward Bayesian mean
        if self.mean_lr < 1.0:
            self.W_mean = (1.0 - self.mean_lr) * M_old + self.mean_lr * M_bayes
        else:
            self.W_mean = M_bayes

        self.W_precision = P_new

    # ------------------------------------------------------------------
    # Hierarchical reward function
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_reward(
        metrics_now: Dict[str, float],
        user_ema_vector: List[float],
        w1: float = 1.0,
        w2: float = 0.7,
        w3: float = 0.4,
        reward_scale: float = 20.0,
    ) -> float:
        """
        Calculates the hierarchical reward based on *deltas vs EMA baseline*.

        metrics_now:
            {
                'accuracy': float in [0,1],
                'wpm': float,          # raw WPM for this snippet
                'iki_cv': float,
                'spike_rate': float,   # e.g., in [0,1]
            }

        user_ema_vector:
            57-dim EMA feature vector.
            Indices:
                - IDX_ACCURACY       -> EMA accuracy
                - IDX_WPM_EFFECTIVE  -> EMA effective WPM (wpm * accuracy)
                - IDX_IKI_CV         -> EMA global IKI CV
                - IDX_SPIKE_RATE     -> EMA spike_rate

        Reward structure (hierarchical):
            Let:
                A = delta accuracy
                C = delta smoothness  (new formula)
                S = delta effective WPM (scaled)

            Then:
                R = w1*A + w2*A*C + w3*A*C*S

            Finally scaled by reward_scale to avoid vanishing updates.
        """
        # 1. Extract baselines from EMA
        if (not user_ema_vector) or len(user_ema_vector) < USER_BASE_DIM:
            # Fallback baselines
            base_acc = 0.90
            base_eff_wpm = 40.0
            base_iki_cv = 0.25 # Neutral value for IKI CV
            base_spike_rate = 0.20 # Neutral value for spike_rate
        else:
            base_acc = float(user_ema_vector[IDX_ACCURACY])
            base_eff_wpm = float(user_ema_vector[IDX_WPM_EFFECTIVE])
            base_iki_cv = float(user_ema_vector[IDX_IKI_CV])
            base_spike_rate = float(user_ema_vector[IDX_SPIKE_RATE])
            
        base_smoothness = 0.5 * (1 / (1 + base_iki_cv)) + 0.5 * (1 - base_spike_rate)

        # 2. Extract current metrics
        now_acc = float(metrics_now.get("accuracy", 0.0))
        raw_wpm = float(metrics_now.get("wpm", 0.0))
        now_iki_cv = float(metrics_now.get('iki_cv', 0.0))
        now_spike_rate = float(metrics_now.get('spike_rate', 0.0))

        # Effective WPM = raw WPM * accuracy
        now_eff_wpm = raw_wpm * now_acc
        now_smoothness = 0.5 * (1 / (1 + now_iki_cv)) + 0.5 * (1 - now_spike_rate)

        # 3. Deltas vs baselines
        delta_A = now_acc - base_acc
        delta_C = now_smoothness - base_smoothness
        delta_S = now_eff_wpm - base_eff_wpm

        # 4. Normalize / clip deltas
        # Accuracy / consistency: typically small deviations
        dA = float(np.clip(delta_A, -0.2, 0.2))
        dC = float(np.clip(delta_C, -0.2, 0.2))

        # Map 10 WPM difference -> ~0.1 (so 100 WPM diff -> 1.0)
        delta_S_scaled = delta_S / 100.0
        dS = float(np.clip(delta_S_scaled, -0.3, 0.3))

        # 5. Hierarchical reward
        term1 = w1 * dA
        term2 = w2 * (dA * dC)
        term3 = w3 * (dA * dC * dS)

        reward = term1 + term2 + term3

        # 6. Scale reward to avoid tiny gradients
        reward *= reward_scale

        # Safety
        if not np.isfinite(reward):
            return 0.0

        return float(reward)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(
                {
                    "mean": self.W_mean,
                    "precision": self.W_precision,
                    "version": self.version,
                    "lambda_prior": self.lambda_prior,
                    "min_var": self.min_var,
                    "max_precision": self.max_precision,
                    "mean_lr": self.mean_lr,
                },
                f,
            )

    def load(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                self.W_mean = data["mean"]
                self.W_precision = data["precision"]
                self.version = data.get("version", 2)
                self.lambda_prior = data.get("lambda_prior", 1.0)
                self.min_var = data.get("min_var", 1e-4)
                self.max_precision = data.get("max_precision", 1e6)
                self.mean_lr = data.get("mean_lr", 1.0)


# Singleton instance
agent = LinTSAgent()
try:
    agent.load()
except Exception as e:
    print(f"Could not load LinTS agent: {e}, starting fresh.")
