import string
import math
from collections import Counter

# -------------------------------------------------------------
# GLOBAL FREQUENCY TABLES (populated by generator)
# -------------------------------------------------------------

BIGRAM_FREQS = {}
TRIGRAM_FREQS = {}
WORD_FREQ = {}

UNKNOWN_BIGRAM = -15.0
UNKNOWN_TRIGRAM = -20.0
UNKNOWN_WORD_FREQ = -12.0


def set_global_ngram_tables(bigrams, trigrams):
    """
    Called by generator/generate.py to inject weighted n-gram tables.
    """
    global BIGRAM_FREQS, TRIGRAM_FREQS
    BIGRAM_FREQS = {str(k): float(v) for k, v in bigrams.items()}
    TRIGRAM_FREQS = {str(k): float(v) for k, v in trigrams.items()}


def set_global_wordfreq(freqs):
    """
    Called by generator/generate.py to inject {word -> zipf} frequency map.
    """
    global WORD_FREQ
    WORD_FREQ = {str(k): float(v) for k, v in freqs.items()}


# -------------------------------------------------------------
# KEYBOARD MAPPING (QWERTY + thumb)
# -------------------------------------------------------------
# (finger_index, hand, row_index)
# Finger: 0-3 (L Pinky->Index), 4-7 (R Index->Pinky), 8 (Thumb)
# Hand: L, R, T
# Row: 0 (Top/Num), 1 (Home), 2 (Bottom) -- Approximation for QWERTY rows
#   Actual QWERTY Rows: 
#   Num Row (1,2...): Row 0
#   Top Row (q,w...): Row 1
#   Home Row (a,s...): Row 2
#   Bot Row (z,x...): Row 3
#   Space: Row 4
#
# Let's align with standard physical rows for 'row_change' detection.
# 'q' is Top (1), 'a' is Home (2), 'z' is Bottom (3).
KEYBOARD_MAP = {
    # Left Hand
    '`': (0, 'L', 0), '1': (0, 'L', 0), '2': (1, 'L', 0), '3': (2, 'L', 0), '4': (3, 'L', 0), '5': (3, 'L', 0),
    'q': (0, 'L', 1), 'w': (1, 'L', 1), 'e': (2, 'L', 1), 'r': (3, 'L', 1), 't': (3, 'L', 1),
    'a': (0, 'L', 2), 's': (1, 'L', 2), 'd': (2, 'L', 2), 'f': (3, 'L', 2), 'g': (3, 'L', 2),
    'z': (0, 'L', 3), 'x': (1, 'L', 3), 'c': (2, 'L', 3), 'v': (3, 'L', 3), 'b': (3, 'L', 3),

    # Right Hand
    '6': (4, 'R', 0), '7': (4, 'R', 0), '8': (5, 'R', 0), '9': (6, 'R', 0), '0': (7, 'R', 0), '-': (7, 'R', 0), '=': (7, 'R', 0),
    'y': (4, 'R', 1), 'u': (5, 'R', 1), 'i': (6, 'R', 1), 'o': (7, 'R', 1), 'p': (7, 'R', 1), '[': (7, 'R', 1), ']': (7, 'R', 1), '\\': (7, 'R', 1),
    'h': (4, 'R', 2), 'j': (5, 'R', 2), 'k': (6, 'R', 2), 'l': (7, 'R', 2), ';': (7, 'R', 2), "'": (7, 'R', 2),
    'n': (4, 'R', 3), 'm': (5, 'R', 3), ',': (6, 'R', 3), '.': (7, 'R', 3), '/': (7, 'R', 3),

    # Thumb
    ' ': (8, 'T', 4)
}

VOWELS = set("aeiou")
LETTER_SET = set(string.ascii_lowercase)

# -------------------------------------------------------------
# MAIN FEATURE EXTRACTOR
# -------------------------------------------------------------

def compute_difficulty_features(text: str) -> dict:
    """
    Computes the 32-dimensional feature set for the snippet.
    """
    if not text:
        return {}
        
    text_lower = text.lower()
    chars = list(text_lower)
    n = len(chars)
    denom = max(1, n)
    trans_denom = max(1, n - 1)

    # ---------------------------------------------------------
    # A. Composition (3)
    # ---------------------------------------------------------
    vowel_count = sum(1 for c in chars if c in VOWELS)
    space_count = sum(1 for c in chars if c == ' ')
    letter_count = sum(1 for c in chars if c in LETTER_SET)
    consonant_count = letter_count - vowel_count
    
    vowel_ratio = vowel_count / denom
    consonant_ratio = consonant_count / denom
    space_ratio = space_count / denom

    # ---------------------------------------------------------
    # B. Hand / Finger / Row Transitions (9)
    # ---------------------------------------------------------
    fingers = []
    hands = []
    rows = []
    
    for c in chars:
        # Default to Thumb Space if unknown
        f, h, r = KEYBOARD_MAP.get(c, (8, 'T', 4))
        fingers.append(f)
        hands.append(h)
        rows.append(r)
        
    l2l = 0
    r2r = 0
    cross = 0
    repeat = 0
    same_finger = 0
    row_changes = 0
    
    # Finger usage counts
    pinky_c = sum(1 for f in fingers if f in [0, 7])
    ring_c = sum(1 for f in fingers if f in [1, 6])
    middle_c = sum(1 for f in fingers if f in [2, 5])
    
    # Transition loop
    for i in range(n - 1):
        h1, h2 = hands[i], hands[i+1]
        f1, f2 = fingers[i], fingers[i+1]
        r1, r2 = rows[i], rows[i+1]
        
        # Hand/Repeat
        if h1 == h2:
            if f1 == f2 and r1 == r2: # Exact key repeat usually, or same key press
                # Wait, 'repeat' usually means same KEY, not just same finger/row?
                # But 'same_finger' captures same finger.
                # Let's assume 'repeat' means same key.
                if chars[i] == chars[i+1]:
                    repeat += 1
                
                # Hand side stats
                if h1 == 'L': l2l += 1
                elif h1 == 'R': r2r += 1
        else:
            # Different hands (ignore thumb logic for cross vs hand for simplicity, or treat T as distinct)
            # Usually Cross is L<->R. T<->L or T<->R is also a hand change.
            cross += 1
            
        # Same Finger (excluding exact repeats, or including? Usually implies difficulty)
        if f1 == f2 and chars[i] != chars[i+1]:
            same_finger += 1
            
        # Row Change
        if r1 != r2:
            row_changes += 1

    l2l_ratio = l2l / trans_denom
    r2r_ratio = r2r / trans_denom
    cross_ratio = cross / trans_denom
    repeat_ratio = repeat / trans_denom
    same_finger_ratio = same_finger / trans_denom
    row_change_ratio = row_changes / trans_denom
    
    pinky_ratio = pinky_c / denom
    ring_ratio = ring_c / denom
    middle_ratio = middle_c / denom

    # ---------------------------------------------------------
    # C. Rollover Opportunity (3)
    # ---------------------------------------------------------
    # Opportunity = Consecutive distinct fingers on same hand, or any cross hand.
    # Excludes same-finger-same-hand.
    l2l_rollover = 0
    r2r_rollover = 0
    cross_rollover = 0
    
    for i in range(n - 1):
        h1, h2 = hands[i], hands[i+1]
        f1, f2 = fingers[i], fingers[i+1]
        
        if h1 == 'L' and h2 == 'L':
            if f1 != f2: l2l_rollover += 1
        elif h1 == 'R' and h2 == 'R':
            if f1 != f2: r2r_rollover += 1
        elif h1 != h2:
            cross_rollover += 1
            
    l2l_rollover_ratio = l2l_rollover / trans_denom
    r2r_rollover_ratio = r2r_rollover / trans_denom
    cross_rollover_ratio = cross_rollover / trans_denom

    # ---------------------------------------------------------
    # D. Flow / Directional Structure (4)
    # ---------------------------------------------------------
    # Flow segment: Sequence of characters typed without direction reversal on the SAME hand, 
    # OR simpler definition: Sequence of distinct fingers?
    # Let's use: Sequence of strictly increasing or strictly decreasing finger indices on the SAME hand.
    # Cross-hand breaks flow segment? Usually yes.
    
    flow_segments_list = []
    current_flow_len = 1
    direction_changes = 0
    
    # Direction: +1 (increasing finger index), -1 (decreasing)
    # Finger indices: L(0-3), R(4-7). 
    # This metric is tricky for cross-hand.
    # Let's simplify: A flow segment is a sequence of non-same-finger keystrokes.
    # Direction change: 
    #  L-Pinky(0) -> L-Index(3) is +direction.
    #  L-Index(3) -> L-Pinky(0) is -direction.
    #  R-Index(4) -> R-Pinky(7) is +direction.
    #  R-Pinky(7) -> R-Index(4) is -direction.
    
    last_dir = 0 # 0 init, 1 up, -1 down
    
    for i in range(n - 1):
        f1, f2 = fingers[i], fingers[i+1]
        h1, h2 = hands[i], hands[i+1]
        
        is_flow_continue = False
        current_dir = 0
        
        if h1 == h2 and f1 != f2:
            # Same hand, distinct fingers
            if f2 > f1: current_dir = 1
            else: current_dir = -1
            
            if last_dir == 0:
                last_dir = current_dir
                is_flow_continue = True
            elif current_dir == last_dir:
                is_flow_continue = True
            else:
                # Direction change
                direction_changes += 1
                last_dir = current_dir
                is_flow_continue = False
        else:
            # Hand change or same finger -> Break flow
            is_flow_continue = False
            last_dir = 0
            
        if is_flow_continue:
            current_flow_len += 1
        else:
            flow_segments_list.append(current_flow_len)
            current_flow_len = 1
            
    flow_segments_list.append(current_flow_len)
    
    flow_segments = len(flow_segments_list)
    longest_flow_segment = max(flow_segments_list) if flow_segments_list else 1
    avg_flow_segment_length = sum(flow_segments_list) / len(flow_segments_list) if flow_segments_list else 1.0
    direction_change_ratio = direction_changes / trans_denom

    # ---------------------------------------------------------
    # E. Repetition Structure (3)
    # ---------------------------------------------------------
    # Max run of specific char
    max_char_run = 1
    current_run = 1
    double_letter_count = 0
    triple_letter_count = 0
    
    for i in range(n - 1):
        if chars[i] == chars[i+1]:
            current_run += 1
        else:
            if current_run > max_char_run: max_char_run = current_run
            if current_run == 2: double_letter_count += 1
            elif current_run == 3: triple_letter_count += 1
            elif current_run > 3: triple_letter_count += 1 # Count 4+ as triple+ for simplicity?
            current_run = 1
            
    if current_run > max_char_run: max_char_run = current_run
    if current_run == 2: double_letter_count += 1
    elif current_run >= 3: triple_letter_count += 1

    # ---------------------------------------------------------
    # F. N-gram / Orthographic Difficulty (7)
    # ---------------------------------------------------------
    bigram_scores = []
    for i in range(n - 1):
        bg = chars[i] + chars[i+1]
        bigram_scores.append(BIGRAM_FREQS.get(bg, UNKNOWN_BIGRAM))
        
    trigram_scores = []
    for i in range(n - 2):
        tg = chars[i] + chars[i+1] + chars[i+2]
        trigram_scores.append(TRIGRAM_FREQS.get(tg, UNKNOWN_TRIGRAM))
        
    bigram_log_freq_avg = sum(bigram_scores) / max(1, len(bigram_scores))
    bigram_log_freq_min = min(bigram_scores) if bigram_scores else UNKNOWN_BIGRAM
    
    trigram_log_freq_avg = sum(trigram_scores) / max(1, len(trigram_scores))
    trigram_log_freq_min = min(trigram_scores) if trigram_scores else UNKNOWN_BIGRAM
    
    # Rare bigram ratio (threshold based on UNKNOWN or low freq)
    # Let's say anything equal to UNKNOWN_BIGRAM is rare
    rare_bigram_count = sum(1 for s in bigram_scores if s <= UNKNOWN_BIGRAM)
    rare_bigram_ratio = rare_bigram_count / max(1, len(bigram_scores))
    
    # Word freq
    words = text_lower.split()
    w_freqs = [WORD_FREQ.get(w, UNKNOWN_WORD_FREQ) for w in words]
    word_log_freq_avg = sum(w_freqs) / max(1, len(w_freqs))
    word_log_freq_min = min(w_freqs) if w_freqs else UNKNOWN_WORD_FREQ

    # ---------------------------------------------------------
    # G. Length / Diversity Meta (3)
    # ---------------------------------------------------------
    snippet_length_chars = float(n)
    unique_char_count = len(set(chars))
    
    # Entropy
    counts = Counter(chars)
    entropy = 0.0
    for k in counts:
        p = counts[k] / n
        entropy -= p * math.log2(p)
    char_entropy = entropy

    # ---------------------------------------------------------
    # RETURN
    # ---------------------------------------------------------
    return {
        # A
        "vowel_ratio": vowel_ratio,
        "consonant_ratio": consonant_ratio,
        "space_ratio": space_ratio,
        # B
        "L2L_ratio": l2l_ratio,
        "R2R_ratio": r2r_ratio,
        "cross_ratio": cross_ratio,
        "repeat_ratio": repeat_ratio,
        "same_finger_ratio": same_finger_ratio,
        "row_change_ratio": row_change_ratio,
        "pinky_ratio": pinky_ratio,
        "ring_ratio": ring_ratio,
        "middle_ratio": middle_ratio,
        # C
        "L2L_rollover_opportunity_ratio": l2l_rollover_ratio,
        "R2R_rollover_opportunity_ratio": r2r_rollover_ratio,
        "cross_rollover_opportunity_ratio": cross_rollover_ratio,
        # D
        "flow_segments": float(flow_segments),
        "longest_flow_segment": float(longest_flow_segment),
        "avg_flow_segment_length": avg_flow_segment_length,
        "direction_change_ratio": direction_change_ratio,
        # E
        "max_char_run": float(max_char_run),
        "double_letter_count": float(double_letter_count),
        "triple_letter_count": float(triple_letter_count),
        # F
        "bigram_log_freq_avg": bigram_log_freq_avg,
        "bigram_log_freq_min": bigram_log_freq_min,
        "trigram_log_freq_avg": trigram_log_freq_avg,
        "trigram_log_freq_min": trigram_log_freq_min,
        "rare_bigram_ratio": rare_bigram_ratio,
        "word_log_freq_avg": word_log_freq_avg,
        "word_log_freq_min": word_log_freq_min,
        # G
        "snippet_length_chars": snippet_length_chars,
        "unique_char_count": float(unique_char_count),
        "char_entropy": char_entropy
    }