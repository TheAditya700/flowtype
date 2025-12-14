"""
Tests for LinTS (Linear Thompson Sampling) Agent.

Covers:
- Context building (get_context)
- Thompson Sampling prediction
- Bayesian update mechanics
- Hierarchical reward calculation
- Save/load persistence
"""

import numpy as np
import pytest
import tempfile
import os

from app.ml.lints_agent import (
    LinTSAgent,
    USER_DIM,
    USER_BASE_DIM,
    SNIPPET_DIM,
    IDX_ACCURACY,
    IDX_IKI_CV,
    IDX_WPM_EFFECTIVE,
    IDX_SPIKE_RATE,
)


class TestLinTSAgentContext:
    """Tests for context/state building."""

    def test_get_context_correct_dimensions(self):
        """Context vector should be 130-dim: EMA(57) + STD(57) + Prev(16)."""
        agent = LinTSAgent()
        user_ema = [0.5] * USER_BASE_DIM
        user_std = [0.1] * USER_BASE_DIM
        prev_embedding = [0.0] * SNIPPET_DIM

        context = agent.get_context(user_ema, user_std, prev_embedding)

        assert context.shape == (USER_DIM,)
        assert context.shape[0] == 130

    def test_get_context_with_none_prev_embedding(self):
        """When prev_embedding is None, should zero-fill."""
        agent = LinTSAgent()
        user_ema = [0.5] * USER_BASE_DIM
        user_std = [0.1] * USER_BASE_DIM

        context = agent.get_context(user_ema, user_std, None)

        assert context.shape == (USER_DIM,)
        # Last 16 dimensions should be zeros
        assert np.allclose(context[-SNIPPET_DIM:], np.zeros(SNIPPET_DIM))

    def test_get_context_resizes_mismatched_inputs(self):
        """Agent should gracefully resize if inputs are slightly off."""
        agent = LinTSAgent()
        # Intentionally wrong sizes
        user_ema = [0.5] * 50  # Should be 57
        user_std = [0.1] * 60  # Should be 57
        prev_embedding = [0.0] * 10  # Should be 16

        context = agent.get_context(user_ema, user_std, prev_embedding)

        # Should still produce correct output dimension
        assert context.shape == (USER_DIM,)

    def test_get_context_preserves_values(self):
        """Context should preserve input values in correct positions."""
        agent = LinTSAgent()
        user_ema = [float(i) for i in range(USER_BASE_DIM)]
        user_std = [float(i + 100) for i in range(USER_BASE_DIM)]
        prev_embedding = [float(i + 200) for i in range(SNIPPET_DIM)]

        context = agent.get_context(user_ema, user_std, prev_embedding)

        # Check EMA section
        assert np.allclose(context[:USER_BASE_DIM], user_ema)
        # Check STD section
        assert np.allclose(context[USER_BASE_DIM : 2 * USER_BASE_DIM], user_std)
        # Check prev_embedding section
        assert np.allclose(context[-SNIPPET_DIM:], prev_embedding)


class TestLinTSAgentPredict:
    """Tests for Thompson Sampling prediction."""

    def test_predict_returns_correct_dimension(self):
        """Predict should return 16-dim query vector."""
        agent = LinTSAgent()
        user_state = np.zeros(USER_DIM, dtype=np.float32)

        query_vector = agent.predict(user_state)

        assert len(query_vector) == SNIPPET_DIM
        assert all(isinstance(x, float) for x in query_vector)

    def test_predict_raises_on_wrong_dimension(self):
        """Predict should raise ValueError for wrong input dimension."""
        agent = LinTSAgent()
        wrong_state = np.zeros(50, dtype=np.float32)

        with pytest.raises(ValueError, match="dim mismatch"):
            agent.predict(wrong_state)

    def test_predict_is_stochastic(self):
        """Thompson Sampling should produce different outputs (exploration)."""
        agent = LinTSAgent()
        user_state = np.ones(USER_DIM, dtype=np.float32)

        # Run multiple predictions
        predictions = [agent.predict(user_state) for _ in range(10)]

        # Not all predictions should be identical (due to sampling)
        unique_predictions = set(tuple(p) for p in predictions)
        assert len(unique_predictions) > 1, "Thompson Sampling should explore"

    def test_predict_deterministic_with_high_precision(self):
        """With very high precision (low variance), predictions should be similar."""
        agent = LinTSAgent(min_var=1e-10)  # Use very low min_var
        agent.W_precision = np.full(
            (SNIPPET_DIM, USER_DIM), 1e10, dtype=np.float32
        )  # Very high precision
        user_state = np.ones(USER_DIM, dtype=np.float32)

        predictions = [agent.predict(user_state) for _ in range(5)]

        # All predictions should be close (relaxed tolerance due to min_var floor)
        for i in range(1, len(predictions)):
            assert np.allclose(predictions[0], predictions[i], atol=1.0)


class TestLinTSAgentUpdate:
    """Tests for Bayesian update mechanics."""

    def test_update_increases_precision(self):
        """After update, precision should increase (learning)."""
        agent = LinTSAgent()
        user_state = np.ones(USER_DIM, dtype=np.float32)
        snippet_vector = np.ones(SNIPPET_DIM, dtype=np.float32)
        reward = 1.0

        old_precision = agent.W_precision.copy()
        agent.update(user_state, snippet_vector, reward)
        new_precision = agent.W_precision

        # Precision should increase where outer product is non-zero
        assert np.all(new_precision >= old_precision)

    def test_update_modifies_mean(self):
        """After update with positive reward, mean should shift."""
        agent = LinTSAgent()
        user_state = np.ones(USER_DIM, dtype=np.float32)
        snippet_vector = np.ones(SNIPPET_DIM, dtype=np.float32)

        old_mean = agent.W_mean.copy()
        agent.update(user_state, snippet_vector, reward=1.0)
        new_mean = agent.W_mean

        # Mean should change
        assert not np.allclose(old_mean, new_mean)

    def test_update_direction_depends_on_reward_sign(self):
        """Positive vs negative rewards should push mean in opposite directions."""
        agent_pos = LinTSAgent()
        agent_neg = LinTSAgent()

        user_state = np.ones(USER_DIM, dtype=np.float32)
        snippet_vector = np.ones(SNIPPET_DIM, dtype=np.float32)

        agent_pos.update(user_state, snippet_vector, reward=1.0)
        agent_neg.update(user_state, snippet_vector, reward=-1.0)

        # Means should differ
        assert not np.allclose(agent_pos.W_mean, agent_neg.W_mean)

    def test_update_raises_on_wrong_dimensions(self):
        """Update should raise ValueError for wrong input dimensions."""
        agent = LinTSAgent()

        with pytest.raises(ValueError, match="user_state dim mismatch"):
            agent.update(np.zeros(50), np.zeros(SNIPPET_DIM), 1.0)

        with pytest.raises(ValueError, match="snippet_vector dim mismatch"):
            agent.update(np.zeros(USER_DIM), np.zeros(10), 1.0)

    def test_precision_clipping(self):
        """Precision should be clipped to max_precision to avoid blowup."""
        agent = LinTSAgent(max_precision=100.0)
        user_state = np.ones(USER_DIM, dtype=np.float32)
        snippet_vector = np.ones(SNIPPET_DIM, dtype=np.float32)

        # Many updates to inflate precision
        for _ in range(1000):
            agent.update(user_state, snippet_vector, reward=1.0)

        # Precision should be capped
        assert np.all(agent.W_precision <= agent.max_precision)


class TestLinTSAgentReward:
    """Tests for hierarchical reward calculation."""

    def test_reward_zero_on_no_improvement(self):
        """If current metrics match EMA baseline, reward should be ~0."""
        user_ema = [0.0] * USER_BASE_DIM
        user_ema[IDX_ACCURACY] = 0.95
        user_ema[IDX_WPM_EFFECTIVE] = 50.0
        user_ema[IDX_IKI_CV] = 0.2
        user_ema[IDX_SPIKE_RATE] = 0.1

        metrics_now = {
            "accuracy": 0.95,
            "wpm": 50.0 / 0.95,  # raw WPM that gives eff_wpm = 50
            "iki_cv": 0.2,
            "spike_rate": 0.1,
        }

        reward = LinTSAgent.calculate_reward(metrics_now, user_ema)

        assert abs(reward) < 1.0, "Reward should be near zero with no improvement"

    def test_reward_positive_on_accuracy_improvement(self):
        """Improved accuracy should yield positive reward."""
        user_ema = [0.0] * USER_BASE_DIM
        user_ema[IDX_ACCURACY] = 0.90
        user_ema[IDX_WPM_EFFECTIVE] = 40.0
        user_ema[IDX_IKI_CV] = 0.3
        user_ema[IDX_SPIKE_RATE] = 0.2

        # Better accuracy
        metrics_now = {
            "accuracy": 0.98,
            "wpm": 50.0,
            "iki_cv": 0.2,
            "spike_rate": 0.1,
        }

        reward = LinTSAgent.calculate_reward(metrics_now, user_ema)

        assert reward > 0, "Improved accuracy should give positive reward"

    def test_reward_negative_on_accuracy_drop(self):
        """Dropped accuracy should yield negative reward."""
        user_ema = [0.0] * USER_BASE_DIM
        user_ema[IDX_ACCURACY] = 0.95
        user_ema[IDX_WPM_EFFECTIVE] = 50.0
        user_ema[IDX_IKI_CV] = 0.2
        user_ema[IDX_SPIKE_RATE] = 0.1

        # Worse accuracy
        metrics_now = {
            "accuracy": 0.80,
            "wpm": 40.0,
            "iki_cv": 0.4,
            "spike_rate": 0.3,
        }

        reward = LinTSAgent.calculate_reward(metrics_now, user_ema)

        assert reward < 0, "Dropped accuracy should give negative reward"

    def test_reward_hierarchical_structure(self):
        """Consistency multiplies with accuracy (hierarchical)."""
        user_ema = [0.0] * USER_BASE_DIM
        user_ema[IDX_ACCURACY] = 0.90
        user_ema[IDX_WPM_EFFECTIVE] = 40.0
        user_ema[IDX_IKI_CV] = 0.5
        user_ema[IDX_SPIKE_RATE] = 0.4

        # Good accuracy AND good consistency
        metrics_both_good = {
            "accuracy": 0.98,
            "wpm": 60.0,
            "iki_cv": 0.1,
            "spike_rate": 0.05,
        }

        # Good accuracy but worse consistency
        metrics_accuracy_only = {
            "accuracy": 0.98,
            "wpm": 60.0,
            "iki_cv": 0.6,
            "spike_rate": 0.5,
        }

        reward_both = LinTSAgent.calculate_reward(metrics_both_good, user_ema)
        reward_acc_only = LinTSAgent.calculate_reward(metrics_accuracy_only, user_ema)

        # Both improvements should give higher reward than accuracy alone
        assert reward_both > reward_acc_only

    def test_reward_handles_empty_ema(self):
        """Should use fallback baselines when EMA is empty."""
        metrics_now = {
            "accuracy": 0.95,
            "wpm": 50.0,
            "iki_cv": 0.2,
            "spike_rate": 0.1,
        }

        # Empty EMA should not raise
        reward = LinTSAgent.calculate_reward(metrics_now, [])

        assert isinstance(reward, float)
        assert np.isfinite(reward)

    def test_reward_finite_on_edge_cases(self):
        """Reward should always be finite, even with edge-case inputs."""
        user_ema = [0.0] * USER_BASE_DIM

        edge_cases = [
            {"accuracy": 0.0, "wpm": 0.0, "iki_cv": 0.0, "spike_rate": 0.0},
            {"accuracy": 1.0, "wpm": 1000.0, "iki_cv": 10.0, "spike_rate": 1.0},
            {"accuracy": 0.5, "wpm": float("inf"), "iki_cv": 0.0, "spike_rate": 0.0},
        ]

        for metrics in edge_cases:
            reward = LinTSAgent.calculate_reward(metrics, user_ema)
            assert np.isfinite(reward), f"Reward not finite for {metrics}"


class TestLinTSAgentPersistence:
    """Tests for save/load functionality."""

    def test_save_and_load_preserves_state(self):
        """Saved agent should restore identical state."""
        import app.ml.lints_agent as agent_module

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "lints_model.pkl")
            original_path = agent_module.MODEL_PATH
            agent_module.MODEL_PATH = model_path

            try:
                agent = LinTSAgent()
                # Modify state
                agent.W_mean = np.random.randn(SNIPPET_DIM, USER_DIM).astype(np.float32)
                agent.W_precision = np.random.uniform(
                    1.0, 10.0, (SNIPPET_DIM, USER_DIM)
                ).astype(np.float32)

                original_mean = agent.W_mean.copy()
                original_precision = agent.W_precision.copy()

                agent.save()

                # Create new agent and load
                new_agent = LinTSAgent()
                new_agent.load()

                assert np.allclose(new_agent.W_mean, original_mean)
                assert np.allclose(new_agent.W_precision, original_precision)
            finally:
                agent_module.MODEL_PATH = original_path

    def test_load_handles_missing_file(self):
        """Load should not raise if file doesn't exist."""
        import app.ml.lints_agent as agent_module

        original_path = agent_module.MODEL_PATH
        agent_module.MODEL_PATH = "/nonexistent/path/model.pkl"

        try:
            agent = LinTSAgent()
            agent.load()  # Should not raise
            # Should keep defaults
            assert agent.W_mean.shape == (SNIPPET_DIM, USER_DIM)
        finally:
            agent_module.MODEL_PATH = original_path


class TestLinTSAgentIntegration:
    """End-to-end integration tests."""

    def test_full_learning_loop(self):
        """Simulate multiple interactions and verify learning."""
        agent = LinTSAgent()

        # Simulate 10 interactions with consistent positive feedback
        for _ in range(10):
            user_ema = np.random.rand(USER_BASE_DIM).tolist()
            user_std = np.random.rand(USER_BASE_DIM).tolist()
            prev_emb = np.random.rand(SNIPPET_DIM).tolist()

            user_state = agent.get_context(user_ema, user_std, prev_emb)
            query = agent.predict(user_state)

            # Simulate choosing a snippet and getting reward
            snippet_vector = np.array(query, dtype=np.float32)  # Ideal snippet
            reward = 1.0

            agent.update(user_state, snippet_vector, reward)

        # After learning, precision should have increased from prior
        assert np.mean(agent.W_precision) > agent.lambda_prior

    def test_agent_adapts_to_user_preferences(self):
        """Agent should learn to favor certain snippet directions."""
        agent = LinTSAgent()

        # Consistently reward snippets in a specific direction
        preferred_direction = np.random.randn(SNIPPET_DIM).astype(np.float32)
        preferred_direction /= np.linalg.norm(preferred_direction)

        user_state = np.ones(USER_DIM, dtype=np.float32)

        for _ in range(50):
            # Positive reward for preferred direction
            agent.update(user_state, preferred_direction, reward=1.0)

            # Negative reward for opposite direction
            agent.update(user_state, -preferred_direction, reward=-0.5)

        # Query vector should align more with preferred direction
        query = agent.predict(user_state)
        query_np = np.array(query)
        query_np /= np.linalg.norm(query_np) + 1e-8

        dot_product = np.dot(query_np, preferred_direction)

        # Should show some preference (dot product > 0)
        # Note: Due to stochasticity, we check mean of multiple samples
        dots = []
        for _ in range(20):
            q = np.array(agent.predict(user_state))
            q /= np.linalg.norm(q) + 1e-8
            dots.append(np.dot(q, preferred_direction))

        mean_dot = np.mean(dots)
        assert (
            mean_dot > 0
        ), f"Agent should prefer rewarded direction, got mean dot={mean_dot}"
