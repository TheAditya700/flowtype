"""
Tests for UserFeatureExtractor.

Covers:
- Hand mapping (QWERTY layout)
- Session update processing
- Keystroke event handling (IKI, rollover, errors)
- Feature vector computation (57-dim)
- Short-term feature computation (12-dim)
- Serialization (to_dict / from_dict)
"""

import numpy as np
import pytest
from typing import Dict, List

from app.ml.user_features import UserFeatureExtractor, KEY_HAND_MAP


class TestHandMapping:
    """Tests for QWERTY hand mapping."""

    def test_left_hand_keys(self):
        """Left-hand keys should map to 'L'."""
        left_keys = ["q", "w", "e", "r", "t", "a", "s", "d", "f", "g", "z", "x", "c", "v", "b"]
        extractor = UserFeatureExtractor()

        for key in left_keys:
            assert extractor.get_hand(key) == "L", f"Key '{key}' should be left hand"
            # Test uppercase too
            assert extractor.get_hand(key.upper()) == "L"

    def test_right_hand_keys(self):
        """Right-hand keys should map to 'R'."""
        right_keys = ["y", "u", "i", "o", "p", "h", "j", "k", "l", "n", "m"]
        extractor = UserFeatureExtractor()

        for key in right_keys:
            assert extractor.get_hand(key) == "R", f"Key '{key}' should be right hand"
            assert extractor.get_hand(key.upper()) == "R"

    def test_unmapped_keys_return_none(self):
        """Keys not in QWERTY letter set should return None."""
        extractor = UserFeatureExtractor()

        assert extractor.get_hand("1") is None
        assert extractor.get_hand(" ") is None
        assert extractor.get_hand("Shift") is None
        assert extractor.get_hand("") is None


class TestSessionUpdate:
    """Tests for update_from_session processing."""

    def _make_session(
        self,
        events: List[Dict],
        wpm: float = 50.0,
        accuracy: float = 0.95,
    ) -> Dict:
        """Helper to create session dict."""
        return {
            "keystroke_events": events,
            "wpm": wpm,
            "accuracy": accuracy,
            "completed": True,
            "quit_progress": 1.0,
        }

    def _make_event(
        self,
        key: str,
        timestamp: int,
        is_correct: bool = True,
        is_backspace: bool = False,
        keyup_timestamp: int = None,
    ) -> Dict:
        """Helper to create keystroke event."""
        return {
            "key": key,
            "timestamp": timestamp,
            "isCorrect": is_correct,
            "isBackspace": is_backspace,
            "keyup_timestamp": keyup_timestamp or timestamp + 50,
        }

    def test_session_count_increments(self):
        """Session count should increment after each update."""
        extractor = UserFeatureExtractor()
        session = self._make_session([
            self._make_event("a", 100),
            self._make_event("b", 200),
        ])

        assert extractor.session_count == 0
        extractor.update_from_session(session)
        assert extractor.session_count == 1
        extractor.update_from_session(session)
        assert extractor.session_count == 2

    def test_wpm_history_populated(self):
        """WPM history should be updated from session."""
        extractor = UserFeatureExtractor()
        session = self._make_session([self._make_event("a", 100)], wpm=60.0)

        extractor.update_from_session(session)

        assert len(extractor.wpm_history) == 1
        assert extractor.wpm_history[0] == 60.0

    def test_wpm_history_caps_at_20(self):
        """WPM history should not exceed 20 entries."""
        extractor = UserFeatureExtractor()

        for i in range(25):
            session = self._make_session([self._make_event("a", 100)], wpm=float(i))
            extractor.update_from_session(session)

        assert len(extractor.wpm_history) == 20
        # Should keep most recent values
        assert extractor.wpm_history[-1] == 24.0

    def test_backspace_counting(self):
        """Backspace events should be counted."""
        extractor = UserFeatureExtractor()
        session = self._make_session([
            self._make_event("a", 100),
            self._make_event("Backspace", 150, is_backspace=True),
            self._make_event("b", 200),
        ])

        extractor.update_from_session(session)

        assert extractor.backspace_count == 1

    def test_error_counting(self):
        """Incorrect keystrokes should be counted as errors."""
        extractor = UserFeatureExtractor()
        session = self._make_session([
            self._make_event("a", 100, is_correct=True),
            self._make_event("b", 200, is_correct=False),
            self._make_event("c", 300, is_correct=False),
            self._make_event("d", 400, is_correct=True),
        ])

        extractor.update_from_session(session)

        assert extractor.total_presses == 4
        assert extractor.total_errors == 2

    def test_burst_error_detection(self):
        """Consecutive errors should be detected as burst errors."""
        extractor = UserFeatureExtractor()
        session = self._make_session([
            self._make_event("a", 100, is_correct=True),
            self._make_event("b", 200, is_correct=False),  # Error 1
            self._make_event("c", 300, is_correct=False),  # Error 2 (burst)
            self._make_event("d", 400, is_correct=False),  # Error 3 (burst continues)
            self._make_event("e", 500, is_correct=True),
        ])

        extractor.update_from_session(session)

        # First burst detection at 2nd error adds 2, then 3rd error adds 1 = 3 total
        assert extractor.burst_error_count == 3

    def test_letter_stats_tracking(self):
        """Per-letter error stats should be tracked."""
        extractor = UserFeatureExtractor()
        session = self._make_session([
            self._make_event("a", 100, is_correct=True),
            self._make_event("a", 200, is_correct=False),
            self._make_event("a", 300, is_correct=True),
            self._make_event("b", 400, is_correct=True),
        ])

        extractor.update_from_session(session)

        assert extractor.char_stats["a"]["presses"] == 3
        assert extractor.char_stats["a"]["errors"] == 1
        assert extractor.char_stats["b"]["presses"] == 1
        assert extractor.char_stats["b"]["errors"] == 0

    def test_iki_stats_global(self):
        """Global IKI (inter-keystroke interval) should be computed."""
        extractor = UserFeatureExtractor()
        session = self._make_session([
            self._make_event("a", 100),
            self._make_event("b", 200),  # IKI = 100
            self._make_event("c", 350),  # IKI = 150
        ])

        extractor.update_from_session(session)

        assert extractor.iki_stats["global"]["count"] == 2
        assert extractor.iki_stats["global"]["sum"] == 250  # 100 + 150

    def test_transition_type_classification(self):
        """Hand transitions should be classified correctly."""
        extractor = UserFeatureExtractor()
        session = self._make_session([
            self._make_event("a", 100),  # Left
            self._make_event("s", 200),  # Left -> L2L
            self._make_event("j", 300),  # Right -> cross
            self._make_event("k", 400),  # Right -> R2R
            self._make_event("k", 500),  # Same -> repeat
        ])

        extractor.update_from_session(session)

        assert extractor.trans_stats["L2L"]["presses"] == 1
        assert extractor.trans_stats["cross"]["presses"] == 1
        assert extractor.trans_stats["R2R"]["presses"] == 1
        assert extractor.trans_stats["repeat"]["presses"] == 1

    def test_rollover_detection(self):
        """Rollover (keydown before previous keyup) should be detected."""
        extractor = UserFeatureExtractor()
        # Rollover: second key pressed before first key released
        session = self._make_session([
            self._make_event("a", 100, keyup_timestamp=200),
            self._make_event("s", 150, keyup_timestamp=250),  # Pressed at 150, before 200
        ])

        extractor.update_from_session(session)

        assert extractor.rollover_count == 1
        assert extractor.rollover_depth_sum == 50  # 200 - 150

    def test_chunking_spike_detection(self):
        """Spikes (long pauses) should be detected for chunking."""
        extractor = UserFeatureExtractor()
        # Create session with one long pause (spike)
        session = self._make_session([
            self._make_event("a", 100),
            self._make_event("b", 200),   # IKI = 100
            self._make_event("c", 300),   # IKI = 100
            self._make_event("d", 600),   # IKI = 300 (spike, > 1.8 * median)
            self._make_event("e", 700),   # IKI = 100
        ])

        extractor.update_from_session(session)

        # Median IKI = 100, threshold = 180. IKI=300 is a spike
        assert extractor.spike_count >= 1

    def test_empty_session_handled(self):
        """Empty session should not cause errors."""
        extractor = UserFeatureExtractor()
        session = self._make_session([])

        # Should not raise
        extractor.update_from_session(session)

        assert extractor.session_count == 0  # No events = no real session


class TestFeatureVector:
    """Tests for compute_user_features (57-dim vector)."""

    def test_feature_vector_dimension(self):
        """Feature vector should be exactly 57 dimensions."""
        extractor = UserFeatureExtractor()
        features = extractor.compute_user_features()

        assert features.shape == (57,)

    def test_feature_vector_initial_values(self):
        """Fresh extractor should have sensible initial values."""
        extractor = UserFeatureExtractor()
        features = extractor.compute_user_features()

        # Accuracy should be 1.0 (no errors)
        assert features[0] == 1.0
        # Error rate should be 0.0
        assert features[1] == 0.0
        # Letter confidences (last 26) should be 0.5 (neutral for unseen)
        letter_confs = features[-26:]
        assert np.allclose(letter_confs, 0.5)

    def test_feature_vector_after_session(self):
        """Feature vector should reflect session data."""
        extractor = UserFeatureExtractor()

        # Simulate a session with some errors
        for _ in range(10):
            extractor.total_presses += 1
        extractor.total_errors = 2  # 80% accuracy

        features = extractor.compute_user_features()

        # Accuracy = 1 - (2/10) = 0.8
        assert abs(features[0] - 0.8) < 0.01
        # Error rate = 2/10 = 0.2
        assert abs(features[1] - 0.2) < 0.01

    def test_feature_vector_wpm(self):
        """WPM features should be computed from history."""
        extractor = UserFeatureExtractor()
        extractor.wpm_history = [40.0, 50.0, 60.0]
        extractor.effective_wpm_history = [38.0, 47.5, 57.0]

        features = extractor.compute_user_features()

        # Find the WPM value in the feature vector (mean of wpm_history)
        expected_wpm_mean = 50.0
        expected_eff_mean = 47.5

        # Check that WPM values appear somewhere in the vector
        # The exact index depends on implementation, so check both values exist
        assert any(abs(f - expected_wpm_mean) < 1.0 for f in features), "WPM mean not found"
        assert any(abs(f - expected_eff_mean) < 1.0 for f in features), "Effective WPM mean not found"

    def test_feature_vector_all_finite(self):
        """All feature values should be finite."""
        extractor = UserFeatureExtractor()

        # Add some data
        extractor.total_presses = 100
        extractor.total_errors = 5
        extractor.wpm_history = [50.0, 55.0]

        features = extractor.compute_user_features()

        assert np.all(np.isfinite(features))


class TestShortTermFeatures:
    """Tests for compute_short_term_features (12-dim)."""

    def _make_event(self, key: str, timestamp: int, is_correct: bool = True) -> Dict:
        return {
            "key": key,
            "timestamp": timestamp,
            "isCorrect": is_correct,
            "isBackspace": False,
        }

    def test_short_term_dimension(self):
        """Short-term feature vector should be 12 dimensions."""
        extractor = UserFeatureExtractor()
        session_data = {
            "keystroke_events": [self._make_event("a", 100)],
            "wpm": 50.0,
            "accuracy": 0.95,
            "completed": True,
            "quit_progress": 1.0,
        }

        features = extractor.compute_short_term_features(session_data)

        assert features.shape == (12,)

    def test_short_term_wpm_normalized(self):
        """WPM should be normalized (wpm / 100)."""
        extractor = UserFeatureExtractor()
        session_data = {
            "keystroke_events": [self._make_event("a", 100)],
            "wpm": 80.0,
            "accuracy": 0.95,
            "completed": True,
            "quit_progress": 1.0,
        }

        features = extractor.compute_short_term_features(session_data)

        # First feature is wpm / 100
        assert abs(features[0] - 0.8) < 0.01

    def test_short_term_accuracy(self):
        """Accuracy should be included directly."""
        extractor = UserFeatureExtractor()
        session_data = {
            "keystroke_events": [self._make_event("a", 100)],
            "wpm": 50.0,
            "accuracy": 0.92,
            "completed": True,
            "quit_progress": 1.0,
        }

        features = extractor.compute_short_term_features(session_data)

        # Second feature is accuracy
        assert abs(features[1] - 0.92) < 0.01

    def test_short_term_quit_detection(self):
        """Quit and ragequit flags should be computed."""
        extractor = UserFeatureExtractor()

        # Normal completion
        completed_session = {
            "keystroke_events": [self._make_event("a", 100)],
            "wpm": 50.0,
            "accuracy": 0.95,
            "completed": True,
            "quit_progress": 1.0,
        }
        features_completed = extractor.compute_short_term_features(completed_session)
        assert features_completed[8] == 0.0  # quit flag
        assert features_completed[9] == 0.0  # ragequit flag

        # Early quit (ragequit: < 15% progress)
        ragequit_session = {
            "keystroke_events": [self._make_event("a", 100)],
            "wpm": 50.0,
            "accuracy": 0.95,
            "completed": False,
            "quit_progress": 0.10,
        }
        features_ragequit = extractor.compute_short_term_features(ragequit_session)
        assert features_ragequit[8] == 1.0  # quit flag
        assert features_ragequit[9] == 1.0  # ragequit flag

    def test_short_term_latency_computed(self):
        """Latency stats should be computed from keystroke timings."""
        extractor = UserFeatureExtractor()
        session_data = {
            "keystroke_events": [
                self._make_event("a", 100),
                self._make_event("b", 200),  # latency = 100
                self._make_event("c", 400),  # latency = 200
            ],
            "wpm": 50.0,
            "accuracy": 0.95,
            "completed": True,
            "quit_progress": 1.0,
        }

        features = extractor.compute_short_term_features(session_data)

        # Features 3, 4, 5 are latency_mean_norm, latency_std_norm, latency_p95_norm
        # Mean latency = 150ms, normalized by 500 = 0.3
        assert features[3] > 0  # Some normalized latency


class TestSerialization:
    """Tests for to_dict / from_dict serialization."""

    def test_round_trip_empty(self):
        """Empty extractor should serialize and deserialize correctly."""
        original = UserFeatureExtractor()
        data = original.to_dict()
        restored = UserFeatureExtractor.from_dict(data)

        assert restored.session_count == original.session_count
        assert restored.total_presses == original.total_presses

    def test_round_trip_with_data(self):
        """Extractor with data should serialize and deserialize correctly."""
        original = UserFeatureExtractor()
        original.total_presses = 100
        original.total_errors = 10
        original.wpm_history = [40.0, 50.0, 60.0]
        original.session_count = 5
        original.rollover_count = 15

        data = original.to_dict()
        restored = UserFeatureExtractor.from_dict(data)

        assert restored.total_presses == 100
        assert restored.total_errors == 10
        assert restored.wpm_history == [40.0, 50.0, 60.0]
        assert restored.session_count == 5
        assert restored.rollover_count == 15

    def test_from_dict_handles_empty_dict(self):
        """from_dict should handle empty dict gracefully."""
        restored = UserFeatureExtractor.from_dict({})

        assert restored.session_count == 0
        assert restored.total_presses == 0

    def test_from_dict_handles_none(self):
        """from_dict should handle None gracefully."""
        restored = UserFeatureExtractor.from_dict(None)

        assert restored.session_count == 0

    def test_feature_vector_same_after_round_trip(self):
        """Feature vector should be identical after serialization round-trip."""
        original = UserFeatureExtractor()
        original.total_presses = 50
        original.total_errors = 5
        original.wpm_history = [45.0, 55.0]
        original.effective_wpm_history = [42.0, 52.0]

        original_features = original.compute_user_features()

        data = original.to_dict()
        restored = UserFeatureExtractor.from_dict(data)
        restored_features = restored.compute_user_features()

        assert np.allclose(original_features, restored_features)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_division_safety(self):
        """Feature computation should handle zero denominators."""
        extractor = UserFeatureExtractor()
        # No presses at all
        extractor.total_presses = 0

        features = extractor.compute_user_features()

        # Should not have NaN or Inf
        assert np.all(np.isfinite(features))

    def test_very_high_error_rate(self):
        """Should handle 100% error rate."""
        extractor = UserFeatureExtractor()
        extractor.total_presses = 10
        extractor.total_errors = 10

        features = extractor.compute_user_features()

        assert features[0] == 0.0  # Accuracy = 0
        assert features[1] == 1.0  # Error rate = 1

    def test_many_sessions_accumulate(self):
        """Stats should accumulate across many sessions."""
        extractor = UserFeatureExtractor()

        for i in range(100):
            extractor.total_presses += 10
            extractor.total_errors += 1

        assert extractor.total_presses == 1000
        assert extractor.total_errors == 100

        features = extractor.compute_user_features()
        assert abs(features[0] - 0.9) < 0.01  # 90% accuracy
