import numpy as np
import statistics
import string
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Standard QWERTY Layout Mapping
KEY_HAND_MAP = {
    'q': 'L', 'w': 'L', 'e': 'L', 'r': 'L', 't': 'L',
    'a': 'L', 's': 'L', 'd': 'L', 'f': 'L', 'g': 'L',
    'z': 'L', 'x': 'L', 'c': 'L', 'v': 'L', 'b': 'L',
    'y': 'R', 'u': 'R', 'i': 'R', 'o': 'R', 'p': 'R',
    'h': 'R', 'j': 'R', 'k': 'R', 'l': 'R', 'n': 'R', 'm': 'R'
}

class UserFeatureExtractor:
    """
    Revised UserFeatureExtractor implementing the 57-feature set:
    - Accuracy (8)
    - Timing Consistency (12)
    - Speed (2)
    - Rollover (5)
    - Chunking (3)
    - Letter Confidence (26)
    """

    def __init__(self):
        # --- A. ACCURACY (Global & Transition-wise) ---
        self.total_presses = 0
        self.total_errors = 0
        self.backspace_count = 0
        self.burst_error_count = 0  # Count of errors that are part of a burst (>=2 consecutive)
        
        # Transition stats: keys are 'L2L', 'R2R', 'cross', 'repeat'
        self.trans_stats = {
            'L2L': {'presses': 0, 'errors': 0},
            'R2R': {'presses': 0, 'errors': 0},
            'cross': {'presses': 0, 'errors': 0},
            'repeat': {'presses': 0, 'errors': 0},
        }

        # --- B. TIMING (Global & Transition-wise) ---
        # Storing sum, sum_sq, count for online variance calculation
        # Keys: 'global', 'L2L', 'R2R', 'cross', 'repeat'
        self.iki_stats = {
            'global': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'L2L': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'R2R': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'cross': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
            'repeat': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
        }
        self.boundary_penalty_sum = 0.0 # Accumulator for boundary pauses

        # --- C. SPEED ---
        self.wpm_history = [] # Store last 20 session WPMs
        self.effective_wpm_history = [] # (WPM * Accuracy)

        # --- D. ROLLOVER ---
        self.rollover_count = 0
        self.rollover_depth_sum = 0.0 # ms of overlap
        self.roll_trans_counts = {
            'L2L': 0,
            'R2R': 0,
            'cross': 0
        }

        # --- E. CHUNKING ---
        self.spike_count = 0
        self.flow_intervals = 0 # Intervals that are NOT spikes
        self.chunk_sum = 0 # Sum of chars per chunk (approx)
        self.chunk_count = 0 # Number of chunks identified

        # --- F. LETTER CONFIDENCE ---
        self.char_stats = {
            c: {'presses': 0, 'errors': 0}
            for c in string.ascii_lowercase
        }
        
        # Meta
        self.session_count = 0
        
        # Short-term history (sequence of 12-dim vectors)
        self.short_term_history = []

    def get_hand(self, key: str) -> Optional[str]:
        return KEY_HAND_MAP.get(key.lower())

    def update_from_session(self, session: Dict):
        """
        Process a completed session to update long-term stats.
        """
        # 1. Compute and store short-term vector for this session
        try:
            st_vec = self.compute_short_term_features(session)
            self.short_term_history.append(st_vec.tolist())
            if len(self.short_term_history) > 50:
                self.short_term_history.pop(0)
        except Exception as e:
            # Fallback or log if needed
            print(f"Error computing short-term features: {e}")

        events = session.get('keystroke_events', [])
        if not events:
            return

        self.session_count += 1
        
        # Update Speed History
        wpm = session.get('wpm', 0.0)
        acc = session.get('accuracy', 1.0) # 0-1
        self.wpm_history.append(wpm)
        self.effective_wpm_history.append(wpm * acc)
        if len(self.wpm_history) > 20:
            self.wpm_history.pop(0)
            self.effective_wpm_history.pop(0)

        # Process Keystrokes
        prev_event = None
        consecutive_errors = 0
        
        # Chunking helpers
        session_ikis = []
        
        for e in events:
            key = e.get('key', '')
            # Skip non-char keys for most stats, but count backspaces
            if e.get('isBackspace', False):
                self.backspace_count += 1
                consecutive_errors = 0 # Reset burst on backspace? Or continue? Usually backspace ends the "typing" of that error.
                prev_event = None # Reset context on backspace to avoid noisy IKIs
                continue
            
            # Filter for valid layout keys for strict IKI/hand analysis
            if len(key) != 1:
                continue
            
            is_correct = e.get('isCorrect', True)
            timestamp = e.get('timestamp', 0)
            keyup_ts = e.get('keyup_timestamp')
            
            # 1. Global Accuracy
            self.total_presses += 1
            if not is_correct:
                self.total_errors += 1
                consecutive_errors += 1
                if consecutive_errors >= 2:
                    # If this is the 2nd error, add 2. If 3rd, add 1 (cumulatively counts all burst errors)
                    if consecutive_errors == 2:
                        self.burst_error_count += 2
                    else:
                        self.burst_error_count += 1
            else:
                consecutive_errors = 0

            # 2. Letter Stats
            char = key.lower()
            if char in self.char_stats:
                self.char_stats[char]['presses'] += 1
                if not is_correct:
                    self.char_stats[char]['errors'] += 1

            # 3. Transitions (require previous event)
            if prev_event and prev_event.get('key'):
                prev_key = prev_event['key']
                prev_ts = prev_event['timestamp']
                prev_keyup = prev_event.get('keyup_timestamp')
                
                iki = max(0, timestamp - prev_ts)
                session_ikis.append(iki)
                
                # Hand Transition Logic
                h1 = self.get_hand(prev_key)
                h2 = self.get_hand(key)
                
                trans_type = None
                if prev_key == key:
                    trans_type = 'repeat'
                elif h1 and h2:
                    if h1 != h2:
                        trans_type = 'cross'
                    elif h1 == 'L':
                        trans_type = 'L2L'
                    else:
                        trans_type = 'R2R'
                
                if trans_type:
                    # Update Transition Accuracy
                    self.trans_stats[trans_type]['presses'] += 1
                    if not is_correct:
                        self.trans_stats[trans_type]['errors'] += 1
                    
                    # Update Transition Timing
                    self.iki_stats[trans_type]['sum'] += iki
                    self.iki_stats[trans_type]['sum_sq'] += (iki ** 2)
                    self.iki_stats[trans_type]['count'] += 1
                
                # Update Global Timing
                self.iki_stats['global']['sum'] += iki
                self.iki_stats['global']['sum_sq'] += (iki ** 2)
                self.iki_stats['global']['count'] += 1
                
                # 4. Rollover
                # Rollover: Current Down < Prev Up
                if prev_keyup and timestamp < prev_keyup:
                    self.rollover_count += 1
                    overlap = prev_keyup - timestamp
                    self.rollover_depth_sum += overlap
                    
                    if trans_type and trans_type in self.roll_trans_counts:
                        self.roll_trans_counts[trans_type] += 1
            
            prev_event = e

        # 5. Chunking (Session Level Processing)
        if session_ikis:
            median_iki = statistics.median(session_ikis)
            # Simple chunk definition: Spike > 1.8 * median
            threshold = 1.8 * median_iki if median_iki > 0 else 300
            
            current_chunk_len = 0
            
            for iki in session_ikis:
                if iki > threshold:
                    self.spike_count += 1
                    # End of chunk
                    if current_chunk_len > 0:
                        self.chunk_count += 1
                        self.chunk_sum += current_chunk_len
                        current_chunk_len = 0 # Reset for new chunk
                else:
                    self.flow_intervals += 1
                    current_chunk_len += 1
            
            # Count the final chunk
            if current_chunk_len > 0:
                self.chunk_count += 1
                self.chunk_sum += current_chunk_len
    
    def compute_short_term_features(self, session_data: dict) -> np.ndarray:
        """
        Compute short-term snippet-level behavioral features (12-dim).
        Used for the RNN/GRU sequence.
        """
        events = session_data.get("keystroke_events", [])
        wpm = session_data.get("wpm", 40.0)
        accuracy = session_data.get("accuracy", 0.85)
        completed = session_data.get("completed", True)
        quit_progress = session_data.get("quit_progress", 1.0)

        # Latency stats
        latencies = []
        prev_ts = 0
        for i, e in enumerate(events):
            if i > 0 and not e.get("isBackspace", False):
                lat = e.get("timestamp", 0) - prev_ts
                if lat > 0: latencies.append(lat)
            prev_ts = e.get("timestamp", 0)
            
        if latencies:
            lat_mean = float(np.mean(latencies))
            lat_std = float(np.std(latencies)) if len(latencies) > 1 else 0.0
            lat_p95 = float(np.percentile(latencies, 95))
        else:
            lat_mean = 300.0; lat_std = 0.0; lat_p95 = 300.0

        latency_mean_norm = lat_mean / 500.0
        latency_std_norm = lat_std / 300.0
        latency_p95_norm = lat_p95 / 800.0

        # Burst / Repetition
        total_errors = 0
        repetition_errors = 0
        burst_errors = 0
        prev_char = None
        prev_correct = True
        current_burst = 0
        
        for e in events:
            if e.get("isBackspace", False): continue
            char = e.get("key", "").lower()
            is_correct = e.get("isCorrect", True)
            
            if not is_correct:
                total_errors += 1
                if prev_char == char: repetition_errors += 1
                if not prev_correct: current_burst += 1
                else: current_burst = 1
            else:
                if current_burst >= 2: burst_errors += current_burst
                current_burst = 0
            
            prev_char = char
            prev_correct = is_correct
            
        if current_burst >= 2: burst_errors += current_burst
        
        rep_err_rate = repetition_errors / max(1, total_errors)
        burst_err_rate = burst_errors / max(1, total_errors)
        error_rate = 1.0 - accuracy

        # Baselines
        wpm_base = np.mean(self.wpm_history) if self.wpm_history else 40.0
        wpm_delta = (wpm - wpm_base) / max(10, wpm_base)
        acc_delta = accuracy - 0.9 # approx

        features = [
            wpm / 100.0, accuracy, error_rate,
            latency_mean_norm, latency_std_norm, latency_p95_norm,
            rep_err_rate, burst_err_rate,
            0.0 if completed else 1.0, # quit
            1.0 if (not completed and quit_progress < 0.15) else 0.0, # ragequit
            wpm_delta, acc_delta
        ]
        return np.array(features, dtype=np.float32)

    def compute_user_features(self) -> np.ndarray:
        """
        Returns the 57-dimensional feature vector.
        """
        # Helpers
        def safe_div(n, d, default=0.0):
            return n / d if d > 0 else default

        def get_iki_stats(key):
            s = self.iki_stats[key]
            if s['count'] == 0:
                return 0.0, 0.0
            mean = s['sum'] / s['count']
            var = (s['sum_sq'] / s['count']) - (mean ** 2)
            std = np.sqrt(max(0, var))
            cv = safe_div(std, mean)
            return mean, cv

        # --- A. ACCURACY (8) ---
        accuracy = 1.0 - safe_div(self.total_errors, self.total_presses)
        error_rate = safe_div(self.total_errors, self.total_presses)
        # KSPC (KeyStrokes Per Character) - requires knowing "target" chars. 
        # total_presses / (total_presses - backspaces). Assuming finalized text length approx = presses - backspaces.
        # Let's approximate KSPC as 1 + error_rate + backspace_rate roughly.
        kspc = safe_div(self.total_presses, (self.total_presses - self.backspace_count - self.total_errors), default=1.0)
        
        backspace_ratio = safe_div(self.backspace_count, self.total_presses)
        burst_error_rate = safe_div(self.burst_error_count, self.total_errors) # Fraction of errors that are bursts
        
        acc_L2L = 1.0 - safe_div(self.trans_stats['L2L']['errors'], self.trans_stats['L2L']['presses'])
        acc_R2R = 1.0 - safe_div(self.trans_stats['R2R']['errors'], self.trans_stats['R2R']['presses'])
        acc_cross = 1.0 - safe_div(self.trans_stats['cross']['errors'], self.trans_stats['cross']['presses'])
        acc_repeat = 1.0 - safe_div(self.trans_stats['repeat']['errors'], self.trans_stats['repeat']['presses'])
        
        # --- B. TIMING (12) ---
        iki_mean, iki_cv = get_iki_stats('global')
        iki_std = iki_mean * iki_cv
        
        iki_L2L_mean, iki_L2L_cv = get_iki_stats('L2L')
        iki_R2R_mean, iki_R2R_cv = get_iki_stats('R2R')
        iki_cross_mean, iki_cross_cv = get_iki_stats('cross')
        iki_repeat_mean, iki_repeat_cv = get_iki_stats('repeat')
        
        boundary_penalty = 0.5 # Placeholder (Requires specific boundary tracking logic not fully in update loop yet)

        # --- C. SPEED (2) ---
        wpm_raw = np.mean(self.wpm_history) if self.wpm_history else 0.0
        wpm_effective = np.mean(self.effective_wpm_history) if self.effective_wpm_history else 0.0

        # --- D. ROLLOVER (5) ---
        # Rate: Rollover counts / Total relevant transitions
        total_transitions = self.iki_stats['global']['count']
        rollover_rate = safe_div(self.rollover_count, total_transitions)
        rollover_depth_mean = safe_div(self.rollover_depth_sum, self.rollover_count)
        
        roll_L2L_rate = safe_div(self.roll_trans_counts['L2L'], self.trans_stats['L2L']['presses'])
        roll_R2R_rate = safe_div(self.roll_trans_counts['R2R'], self.trans_stats['R2R']['presses'])
        roll_cross_rate = safe_div(self.roll_trans_counts['cross'], self.trans_stats['cross']['presses'])

        # --- E. CHUNKING (3) ---
        total_intervals = self.spike_count + self.flow_intervals
        spike_rate = safe_div(self.spike_count, total_intervals)
        flow_ratio = 1.0 - spike_rate
        avg_chars_per_chunk = safe_div(self.chunk_sum, self.chunk_count, default=1.0)

        # --- F. LETTER CONFIDENCE (26) ---
        letter_confs = []
        for c in string.ascii_lowercase:
            s = self.char_stats[c]
            # Confidence: 1 - error_rate, damped by low sample size
            presses = s['presses']
            errors = s['errors']
            if presses < 5:
                conf = 0.5 # Neutral confidence for unseen
            else:
                conf = 1.0 - (errors / presses)
            letter_confs.append(conf)

        # Vector Assembly
        features = [
            # Accuracy (8)
            accuracy, error_rate, kspc, backspace_ratio, burst_error_rate,
            acc_L2L, acc_R2R, acc_cross, acc_repeat,
            
            # Timing (12)
            iki_mean, iki_std, iki_cv,
            iki_L2L_mean, iki_L2L_cv,
            iki_R2R_mean, iki_R2R_cv,
            iki_cross_mean, iki_cross_cv,
            iki_repeat_mean, iki_repeat_cv,
            boundary_penalty,
            
            # Speed (2)
            wpm_raw, wpm_effective,
            
            # Rollover (5)
            rollover_rate, rollover_depth_mean,
            roll_L2L_rate, roll_R2R_rate, roll_cross_rate,
            
            # Chunking (3)
            spike_rate, flow_ratio, avg_chars_per_chunk
        ] + letter_confs # (26)
        
        return np.array(features, dtype=np.float32)

    def to_dict(self) -> Dict:
        """Serialize state."""
        return {
            'total_presses': self.total_presses,
            'total_errors': self.total_errors,
            'backspace_count': self.backspace_count,
            'burst_error_count': self.burst_error_count,
            'trans_stats': self.trans_stats,
            'iki_stats': self.iki_stats,
            'boundary_penalty_sum': self.boundary_penalty_sum,
            'wpm_history': self.wpm_history,
            'effective_wpm_history': self.effective_wpm_history,
            'rollover_count': self.rollover_count,
            'rollover_depth_sum': self.rollover_depth_sum,
            'roll_trans_counts': self.roll_trans_counts,
            'spike_count': self.spike_count,
            'flow_intervals': self.flow_intervals,
            'chunk_sum': self.chunk_sum,
            'chunk_count': self.chunk_count,
            'char_stats': self.char_stats,
            'session_count': self.session_count,
            'short_term_history': self.short_term_history
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserFeatureExtractor":
        """Deserialize state."""
        e = cls()
        if not data:
            return e
            
        e.total_presses = data.get('total_presses', 0)
        e.total_errors = data.get('total_errors', 0)
        e.backspace_count = data.get('backspace_count', 0)
        e.burst_error_count = data.get('burst_error_count', 0)
        
        e.trans_stats = data.get('trans_stats', e.trans_stats)
        e.iki_stats = data.get('iki_stats', e.iki_stats)
        e.boundary_penalty_sum = data.get('boundary_penalty_sum', 0.0)
        
        e.wpm_history = data.get('wpm_history', [])
        e.effective_wpm_history = data.get('effective_wpm_history', [])
        
        e.rollover_count = data.get('rollover_count', 0)
        e.rollover_depth_sum = data.get('rollover_depth_sum', 0.0)
        e.roll_trans_counts = data.get('roll_trans_counts', e.roll_trans_counts)
        
        e.spike_count = data.get('spike_count', 0)
        e.flow_intervals = data.get('flow_intervals', 0)
        e.chunk_sum = data.get('chunk_sum', 0)
        e.chunk_count = data.get('chunk_count', 0)
        
        e.char_stats = data.get('char_stats', e.char_stats)
        e.session_count = data.get('session_count', 0)
        e.short_term_history = data.get('short_term_history', [])
        
        return e