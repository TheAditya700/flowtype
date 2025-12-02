import numpy as np
import statistics
import string
from collections import defaultdict
from typing import Dict, List, Tuple


class UserFeatureExtractor:
    """
    Clean, layout-agnostic user feature extractor.
    Keeps only skill-based signals:
        - WPM / accuracy performance
        - per-letter confidence (26 features)
        - repetition difficulty
        - n-gram difficulty
        - word difficulty
        - difficulty curve
        - behavioral stability
    """

    def __init__(self):
        # Global performance stats
        self.wpm_history = []
        self.accuracy_history = []
        self.session_count = 0

        # Per-letter stats: {char -> {'presses': N, 'errors': N}}
        self.char_stats = {
            c: {'presses': 0, 'errors': 0}
            for c in string.ascii_lowercase
        }

        # Bigram stats
        self.bigram_stats = defaultdict(lambda: {'presses': 0, 'errors': 0})

        # Trigram stats
        self.trigram_stats = defaultdict(lambda: {'presses': 0, 'errors': 0})

        # Repetition difficulty
        self.double_letter_presses = 0
        self.double_letter_errors = 0
        self.triple_letter_presses = 0
        self.triple_letter_errors = 0

        # Difficulty buckets (1–10)
        self.difficulty_bucket_stats = {
            i: {'total': 0, 'wpm_sum': 0, 'accuracy_sum': 0}
            for i in range(1, 11)
        }

        # Behavioral
        self.snippet_completions = 0
        self.snippet_quits = 0
        self.ragequits = 0
        self.session_lengths = []

        # Word difficulty (rare/common)
        self.rare_word_presses = 0
        self.rare_word_errors = 0
        self.common_word_presses = 0
        self.common_word_errors = 0

        # Short-term history (sequence of 12-dim vectors)
        self.short_term_history = []

    # ------------------------------------------------------------------
    # Updating user stats
    # ------------------------------------------------------------------

    def update_from_session(self, session: Dict):
        """
        session:
        {
            'keystroke_events': [...],
            'wpm': float,
            'accuracy': float,
            'snippet_text': str,
            'snippet_difficulty': float,
            'completed': bool,
            'quit_progress': float
        }
        """
        # 1. Compute and store short-term vector for this session
        st_vec = self.compute_short_term_features(session)
        # Store as list for JSON serialization
        self.short_term_history.append(st_vec.tolist())
        
        # Keep history reasonable size (e.g. last 50)
        if len(self.short_term_history) > 50:
            self.short_term_history.pop(0)

        self.session_count += 1
        self.wpm_history.append(session['wpm'])
        self.accuracy_history.append(session['accuracy'])
        self.session_lengths.append(len(session['keystroke_events']))

        # Completion behavior
        if session.get('completed', True):
            self.snippet_completions += 1
        else:
            self.snippet_quits += 1
            if session.get('quit_progress', 1.0) < 0.15:
                self.ragequits += 1

        # Difficulty bucket stats
        bucket = int(session['snippet_difficulty'])
        stats = self.difficulty_bucket_stats[bucket]
        stats['total'] += 1
        stats['wpm_sum'] += session['wpm']
        stats['accuracy_sum'] += session['accuracy']

        events = session['keystroke_events']
        prev_char = None
        prev_prev_char = None

        # Process keystrokes
        for e in events:
            if e.get('is_backspace', False):
                continue

            char = e['key'].lower()
            if char not in string.ascii_lowercase:
                continue

            is_correct = e.get('is_correct', True)

            # Per-letter stats
            self.char_stats[char]['presses'] += 1
            if not is_correct:
                self.char_stats[char]['errors'] += 1

            # Bigram stats
            if prev_char is not None:
                bigram = prev_char + char
                self.bigram_stats[bigram]['presses'] += 1
                if not is_correct:
                    self.bigram_stats[bigram]['errors'] += 1

            # Trigram stats
            if prev_prev_char is not None:
                trigram = prev_prev_char + prev_char + char
                self.trigram_stats[trigram]['presses'] += 1
                if not is_correct:
                    self.trigram_stats[trigram]['errors'] += 1

            # Repetition stats
            if prev_char == char:
                self.double_letter_presses += 1
                if not is_correct:
                    self.double_letter_errors += 1

                if prev_prev_char == char:
                    self.triple_letter_presses += 1
                    if not is_correct:
                        self.triple_letter_errors += 1

            prev_prev_char = prev_char
            prev_char = char

    # ------------------------------------------------------------------
    # Compute feature vector
    # ------------------------------------------------------------------

    def compute_user_features(self, window='recent') -> np.ndarray:
        """
        Returns a feature vector including:
            - performance stats
            - per-letter confidence (26)
            - repetition difficulty
            - n-gram difficulty
            - word difficulty
            - difficulty curve
            - behavioral traits
        """

        # Select window
        if window == 'recent':
            wpm_data = self.wpm_history[-20:]
            acc_data = self.accuracy_history[-20:]
        else:
            wpm_data = self.wpm_history
            acc_data = self.accuracy_history

        # 1. PERFORMANCE ------------------------------------------------
        wpm_avg = np.mean(wpm_data) if wpm_data else 40.0
        wpm_best = max(wpm_data) if wpm_data else 40.0
        wpm_std = np.std(wpm_data) if len(wpm_data) > 1 else 10.0

        acc_avg = np.mean(acc_data) if acc_data else 0.85
        acc_std = np.std(acc_data) if len(acc_data) > 1 else 0.1

        # 2. PER-LETTER CONFIDENCE (26) --------------------------------
        letter_conf = []
        for c in string.ascii_lowercase:
            presses = self.char_stats[c]['presses']
            errors = self.char_stats[c]['errors']
            conf = 1 - (errors / presses) if presses > 10 else 0.5
            letter_conf.append(conf)

        # 3. REPETITIONS ------------------------------------------------
        double_conf = 1 - (self.double_letter_errors / self.double_letter_presses) \
            if self.double_letter_presses > 0 else 0.7

        triple_conf = 1 - (self.triple_letter_errors / self.triple_letter_presses) \
            if self.triple_letter_presses > 0 else 0.7

        # 4. N-GRAM TOLERANCE ------------------------------------------
        # Approximation: rare = low frequency in your corpus
        rare_bigram_errors = sum(v['errors'] for k, v in self.bigram_stats.items()
                                 if len(k) == 2 and v['presses'] < 5)
        rare_bigram_presses = sum(v['presses'] for k, v in self.bigram_stats.items()
                                  if len(k) == 2 and v['presses'] < 5)

        rare_bigram_conf = 1 - (rare_bigram_errors / rare_bigram_presses) \
            if rare_bigram_presses > 0 else 0.7

        rare_trigram_errors = sum(v['errors'] for k, v in self.trigram_stats.items()
                                  if len(k) == 3 and v['presses'] < 5)
        rare_trigram_presses = sum(v['presses'] for k, v in self.trigram_stats.items()
                                   if len(k) == 3 and v['presses'] < 5)

        rare_trigram_conf = 1 - (rare_trigram_errors / rare_trigram_presses) \
            if rare_trigram_presses > 0 else 0.7

        # 5. DIFFICULTY CURVE ------------------------------------------
        comfort = 0.5
        struggle = 1.0

        best_acc = 0
        for diff, stats in self.difficulty_bucket_stats.items():
            if stats['total'] > 0:
                avg_acc = stats['accuracy_sum'] / stats['total']
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    comfort = diff / 10.0

        for diff, stats in sorted(self.difficulty_bucket_stats.items()):
            if stats['total'] > 0:
                avg_acc = stats['accuracy_sum'] / stats['total']
                if avg_acc < 0.75:  # struggle threshold
                    struggle = diff / 10.0
                    break

        # 6. BEHAVIOR ---------------------------------------------------
        total_snippets = self.snippet_completions + self.snippet_quits
        completion_rate = self.snippet_completions / total_snippets if total_snippets > 0 else 1.0
        ragequit_rate = self.ragequits / max(1, self.snippet_quits)

        avg_session_length = np.mean(self.session_lengths) if self.session_lengths else 0.0

        # Assemble flat feature vector
        vec = np.array(
            [
                # Performance
                wpm_avg / 100.0,
                wpm_best / 100.0,
                wpm_std / 50.0,
                acc_avg,
                acc_std,

                # Repetitions
                double_conf,
                triple_conf,

                # N-gram tolerance
                rare_bigram_conf,
                rare_trigram_conf,

                # Difficulty curve
                comfort,
                struggle,

                # Behavior
                completion_rate,
                1 - ragequit_rate,
                avg_session_length / 200.0,
            ] +
            letter_conf,     # 26-dim
            dtype=np.float32
        )

        # Normalize to unit vector
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec

    def get_difficulty_boundaries(self) -> Tuple[float, float]:
        """
        Returns (comfort, struggle) difficulty levels (0.0-1.0) based on history.
        """
        comfort = 0.5
        struggle = 1.0

        best_acc = 0
        for diff, stats in self.difficulty_bucket_stats.items():
            if stats['total'] > 0:
                avg_acc = stats['accuracy_sum'] / stats['total']
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    comfort = diff / 10.0

        for diff, stats in sorted(self.difficulty_bucket_stats.items()):
            if stats['total'] > 0:
                avg_acc = stats['accuracy_sum'] / stats['total']
                if avg_acc < 0.75:  # struggle threshold
                    struggle = diff / 10.0
                    break
        
        return comfort, struggle

    def feature_dim(self) -> int:
        return 14 + 26

    def compute_short_term_features(self, session_data: dict) -> np.ndarray:
        """
        Compute short-term snippet-level behavioral features
        from a single session (one snippet attempt).

        Returns: np.ndarray of shape (12,)
        """

        events = session_data.get("keystroke_events", [])
        wpm = session_data.get("wpm", 40.0)
        accuracy = session_data.get("accuracy", 0.85)  # assume 0–1; adjust if you're using 0–100
        completed = session_data.get("completed", True)
        quit_progress = session_data.get("quit_progress", 1.0)

        # ----------------------------
        # Latency stats
        # ----------------------------
        latencies = [
            e.get("latency", 0.0)
            for e in events
            if not e.get("is_backspace", False) and e.get("latency", 0.0) > 0
        ]

        if latencies:
            lat_mean = float(np.mean(latencies))
            lat_std = float(np.std(latencies)) if len(latencies) > 1 else 0.0
            lat_p95 = float(np.percentile(latencies, 95))
        else:
            lat_mean = 300.0
            lat_std = 0.0
            lat_p95 = 300.0

        # normalize roughly to [0,1]-ish ranges
        latency_mean_norm = lat_mean / 500.0
        latency_std_norm = lat_std / 300.0
        latency_p95_norm = lat_p95 / 800.0

        # ----------------------------
        # Repetition & burst errors
        # ----------------------------
        total_errors = 0
        repetition_errors = 0
        burst_errors = 0

        prev_char = None
        prev_correct = True
        current_burst_len = 0

        for e in events:
            if e.get("is_backspace", False):
                continue

            char = e.get("key", "").lower()
            is_correct = e.get("is_correct", True)

            if not is_correct:
                total_errors += 1

            # repetition: same char as previous and incorrect
            if prev_char is not None and char == prev_char and not is_correct:
                repetition_errors += 1

            # error bursts: consecutive incorrect keypresses
            if not is_correct:
                if not prev_correct:
                    current_burst_len += 1
                else:
                    current_burst_len = 1
            else:
                if current_burst_len >= 2:
                    burst_errors += current_burst_len
                current_burst_len = 0

            prev_char = char
            prev_correct = is_correct

        # if a burst was ongoing at the end
        if current_burst_len >= 2:
            burst_errors += current_burst_len

        total_errors = max(total_errors, 1)  # avoid division by zero

        repetition_error_rate = repetition_errors / total_errors if total_errors > 0 else 0.0
        burst_error_rate = burst_errors / total_errors if total_errors > 0 else 0.0

        # ----------------------------
        # Baseline deltas (vs recent history)
        # ----------------------------
        if self.wpm_history:
            recent_wpm = self.wpm_history[-20:] if len(self.wpm_history) > 20 else self.wpm_history
            wpm_baseline = float(np.mean(recent_wpm))
        else:
            wpm_baseline = 40.0

        if self.accuracy_history:
            recent_acc = self.accuracy_history[-20:] if len(self.accuracy_history) > 20 else self.accuracy_history
            acc_baseline = float(np.mean(recent_acc))
        else:
            acc_baseline = 0.85

        # assume accuracy is 0–1; adjust denominator if you store 0–100
        wpm_delta = (wpm - wpm_baseline) / max(10.0, wpm_baseline)
        acc_delta = (accuracy - acc_baseline)  # already ~[-1,1] if in [0,1]

        # ----------------------------
        # Quit / ragequit
        # ----------------------------
        quit_flag = 0.0 if completed else 1.0
        ragequit_flag = 1.0 if (not completed and quit_progress < 0.15) else 0.0

        # ----------------------------
        # Assemble 12-dim short-term vector
        # ----------------------------
        error_rate = 1.0 - accuracy  # if accuracy in [0,1]

        features = np.array([
            # absolute performance
            wpm / 100.0,          # wpm_norm
            accuracy,             # accuracy
            error_rate,           # error_rate

            # smoothness / hesitation
            latency_mean_norm,
            latency_std_norm,
            latency_p95_norm,

            # structure-specific struggle
            repetition_error_rate,
            burst_error_rate,

            # frustration signals
            quit_flag,
            ragequit_flag,

            # relative to baseline
            wpm_delta,
            acc_delta,
        ], dtype=np.float32)

        return features


    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    
    def to_dict(self) -> Dict:
        """
        Serializes the extractor state to a dictionary for DB storage.
        """
        return {
            "wpm_history": self.wpm_history,
            "accuracy_history": self.accuracy_history,
            "session_count": self.session_count,
            "char_stats": self.char_stats,
            "bigram_stats": {k: v for k, v in self.bigram_stats.items()}, # default_dict to dict
            "trigram_stats": {k: v for k, v in self.trigram_stats.items()},
            "double_letter_presses": self.double_letter_presses,
            "double_letter_errors": self.double_letter_errors,
            "triple_letter_presses": self.triple_letter_presses,
            "triple_letter_errors": self.triple_letter_errors,
            "difficulty_bucket_stats": self.difficulty_bucket_stats,
            "snippet_completions": self.snippet_completions,
            "snippet_quits": self.snippet_quits,
            "ragequits": self.ragequits,
            "session_lengths": self.session_lengths,
            "rare_word_presses": self.rare_word_presses,
            "rare_word_errors": self.rare_word_errors,
            "common_word_presses": self.common_word_presses,
            "common_word_errors": self.common_word_errors,
            "short_term_history": self.short_term_history,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserFeatureExtractor":
        """
        Restores extractor state from a dictionary.
        """
        extractor = cls()
        if not data:
            return extractor

        extractor.wpm_history = data.get("wpm_history", [])
        extractor.accuracy_history = data.get("accuracy_history", [])
        extractor.session_count = data.get("session_count", 0)
        
        # Char stats
        saved_char_stats = data.get("char_stats", {})
        for c, stats in saved_char_stats.items():
            if c in extractor.char_stats:
                extractor.char_stats[c] = stats

        # N-grams (convert back to defaultdict)
        extractor.bigram_stats = defaultdict(lambda: {'presses': 0, 'errors': 0}, data.get("bigram_stats", {}))
        extractor.trigram_stats = defaultdict(lambda: {'presses': 0, 'errors': 0}, data.get("trigram_stats", {}))

        extractor.double_letter_presses = data.get("double_letter_presses", 0)
        extractor.double_letter_errors = data.get("double_letter_errors", 0)
        extractor.triple_letter_presses = data.get("triple_letter_presses", 0)
        extractor.triple_letter_errors = data.get("triple_letter_errors", 0)

        # Difficulty buckets (ensure int keys)
        saved_buckets = data.get("difficulty_bucket_stats", {})
        for k, v in saved_buckets.items():
            extractor.difficulty_bucket_stats[int(k)] = v

        extractor.snippet_completions = data.get("snippet_completions", 0)
        extractor.snippet_quits = data.get("snippet_quits", 0)
        extractor.ragequits = data.get("ragequits", 0)
        extractor.session_lengths = data.get("session_lengths", [])
        
        extractor.rare_word_presses = data.get("rare_word_presses", 0)
        extractor.rare_word_errors = data.get("rare_word_errors", 0)
        extractor.common_word_presses = data.get("common_word_presses", 0)
        extractor.common_word_errors = data.get("common_word_errors", 0)
        
        extractor.short_term_history = data.get("short_term_history", [])

        return extractor
    
    
