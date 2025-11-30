import string

# -------------------------------------------------------------
# GLOBAL FREQUENCY TABLES (populated by generator)
# -------------------------------------------------------------

# These remain empty at import time.
# They will be filled via set_global_ngram_tables() and set_global_wordfreq().
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

KEYBOARD_MAP = {
    # left hand
    'q': (0, "L", 0), 'a': (0, "L", 1), 'z': (0, "L", 2),
    'w': (1, "L", 0), 's': (1, "L", 1), 'x': (1, "L", 2),
    'e': (2, "L", 0), 'd': (2, "L", 1), 'c': (2, "L", 2),
    'r': (3, "L", 0), 'f': (3, "L", 1), 'v': (3, "L", 2),
    't': (3, "L", 0), 'g': (3, "L", 1), 'b': (3, "L", 2),

    # right hand
    'y': (4, "R", 0), 'h': (4, "R", 1), 'n': (4, "R", 2),
    'u': (5, "R", 0), 'j': (5, "R", 1), 'm': (5, "R", 2),
    'i': (6, "R", 0), 'k': (6, "R", 1),
    'o': (7, "R", 0), 'l': (7, "R", 1),
    'p': (7, "R", 0),

    # thumb (space)
    ' ': (8, "T", 1),
}

VOWELS = set("aeiou")
LETTER_SET = set(string.ascii_lowercase)


# -------------------------------------------------------------
# MAIN FEATURE EXTRACTOR
# -------------------------------------------------------------

def compute_difficulty_features(text: str) -> dict:
    """
    Computes ergonomic + orthographic + n-gram + word-level difficulty metrics.
    Returns a dict of scalar features.
    """

    text = text.strip().lower()
    chars = list(text)
    n = len(chars)

    if n < 2:
        return {"error": "too short"}

    # ---------------------------------------------------------
    # Character type ratios
    # ---------------------------------------------------------
    letter_count = sum(c in LETTER_SET for c in chars)
    vowel_count = sum(c in VOWELS for c in chars)
    digit_count = sum(c.isdigit() for c in chars)
    punct_count = sum(c in string.punctuation for c in chars)
    space_count = chars.count(" ")

    denom = max(1, n)

    vowel_ratio = vowel_count / denom
    consonant_ratio = (letter_count - vowel_count) / denom
    digit_ratio = digit_count / denom
    punct_ratio = punct_count / denom
    space_ratio = space_count / denom

    # ---------------------------------------------------------
    # Finger/hand/row sequences
    # ---------------------------------------------------------
    finger_seq, hand_seq, row_seq = [], [], []

    for c in chars:
        finger, hand, row = KEYBOARD_MAP.get(c, (8, "T", 1))  # fallback to thumb
        finger_seq.append(finger)
        hand_seq.append(hand)
        row_seq.append(row)

    # ---------------------------------------------------------
    # Same-finger, hand-alternation, row transitions
    # ---------------------------------------------------------
    same_finger = sum(finger_seq[i] == finger_seq[i+1] for i in range(n - 1))
    hand_alts = sum(hand_seq[i] != hand_seq[i+1] for i in range(n - 1))
    row_changes = sum(row_seq[i] != row_seq[i+1] for i in range(n - 1))

    pinky_count = sum(1 for f in finger_seq if f in (0, 7))
    ring_count = sum(1 for f in finger_seq if f in (1, 6))

    # pinky runs
    pinky_runs = 0
    run = 0
    for f in finger_seq:
        if f in (0, 7):
            run += 1
        else:
            if run > 1:
                pinky_runs += 1
            run = 0
    if run > 1:
        pinky_runs += 1

    same_finger_ratio = same_finger / (n - 1)
    hand_alt_ratio = hand_alts / (n - 1)
    row_change_ratio = row_changes / (n - 1)
    pinky_ratio = pinky_count / denom
    ring_ratio = ring_count / denom

    # ---------------------------------------------------------
    # FLOW SEGMENTS
    # ---------------------------------------------------------
    flow_segments = 1
    longest_flow = 1
    flow_run = 1

    for i in range(n - 1):
        if finger_seq[i] != finger_seq[i+1]:
            flow_run += 1
        else:
            longest_flow = max(longest_flow, flow_run)
            flow_run = 1
            flow_segments += 1

    longest_flow = max(longest_flow, flow_run)
    avg_flow_len = longest_flow / max(1, flow_segments)

    # ---------------------------------------------------------
    # Directionality
    # ---------------------------------------------------------
    left_to_right = 0
    right_to_left = 0
    direction_changes = 0
    prev_dir = None

    for i in range(n - 1):
        if finger_seq[i] == finger_seq[i+1]:
            continue
        d = 1 if finger_seq[i+1] > finger_seq[i] else -1
        if d == 1:
            left_to_right += 1
        else:
            right_to_left += 1
        if prev_dir is not None and d != prev_dir:
            direction_changes += 1
        prev_dir = d

    total_dirs = max(1, left_to_right + right_to_left)
    left_to_right_ratio = left_to_right / total_dirs
    right_to_left_ratio = right_to_left / total_dirs
    direction_change_ratio = direction_changes / total_dirs

    # ---------------------------------------------------------
    # Repetition difficulty
    # ---------------------------------------------------------
    runs = []
    run = 1
    for i in range(n - 1):
        if chars[i] == chars[i+1]:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)

    max_char_run = max(runs)
    avg_char_run = sum(runs) / len(runs)
    double_letter_count = sum(1 for r in runs if r == 2)
    triple_letter_count = sum(1 for r in runs if r == 3)

    # ---------------------------------------------------------
    # N-gram scores
    # ---------------------------------------------------------
    bigram_scores = [
        BIGRAM_FREQS.get(chars[i] + chars[i+1], UNKNOWN_BIGRAM)
        for i in range(n - 1)
    ]

    trigram_scores = [
        TRIGRAM_FREQS.get(chars[i] + chars[i+1] + chars[i+2], UNKNOWN_TRIGRAM)
        for i in range(n - 2)
    ]

    total_ngrams = (len(bigram_scores) + len(trigram_scores)) or 1
    rare_ngrams = (
        sum(1 for s in bigram_scores if s == UNKNOWN_BIGRAM)
        + sum(1 for s in trigram_scores if s == UNKNOWN_TRIGRAM)
    )

    bigram_log_freq_avg = sum(bigram_scores) / max(1, len(bigram_scores))
    bigram_log_freq_min = min(bigram_scores) if bigram_scores else 0.0

    trigram_log_freq_avg = sum(trigram_scores) / max(1, len(trigram_scores))
    trigram_log_freq_min = min(trigram_scores) if trigram_scores else 0.0

    rare_ngram_ratio = rare_ngrams / total_ngrams

    # ---------------------------------------------------------
    # WORD-LEVEL FREQS
    # ---------------------------------------------------------
    word_freqs = [WORD_FREQ.get(w, UNKNOWN_WORD_FREQ) for w in text.split()]

    if word_freqs:
        word_log_freq_avg = sum(word_freqs) / len(word_freqs)
        word_log_freq_min = min(word_freqs)
        word_log_freq_max = max(word_freqs)
    else:
        word_log_freq_avg = word_log_freq_min = word_log_freq_max = UNKNOWN_WORD_FREQ

    # ---------------------------------------------------------
    # RETURN FULL FEATURE SET
    # ---------------------------------------------------------
    return {
        "vowel_ratio": vowel_ratio,
        "consonant_ratio": consonant_ratio,
        "digit_ratio": digit_ratio,
        "punct_ratio": punct_ratio,
        "space_ratio": space_ratio,

        "same_finger_ratio": same_finger_ratio,
        "hand_alt_ratio": hand_alt_ratio,
        "row_change_ratio": row_change_ratio,
        "pinky_ratio": pinky_ratio,
        "ring_ratio": ring_ratio,
        "pinky_runs": pinky_runs,

        "flow_segments": flow_segments,
        "longest_flow_segment": longest_flow,
        "avg_flow_segment_length": avg_flow_len,

        "left_to_right_ratio": left_to_right_ratio,
        "right_to_left_ratio": right_to_left_ratio,
        "direction_changes": direction_changes,
        "direction_change_ratio": direction_change_ratio,

        "max_char_run": max_char_run,
        "avg_char_run": avg_char_run,
        "double_letter_count": double_letter_count,
        "triple_letter_count": triple_letter_count,

        "bigram_log_freq_avg": bigram_log_freq_avg,
        "bigram_log_freq_min": bigram_log_freq_min,
        "trigram_log_freq_avg": trigram_log_freq_avg,
        "trigram_log_freq_min": trigram_log_freq_min,
        "rare_ngram_ratio": rare_ngram_ratio,

        "word_log_freq_avg": word_log_freq_avg,
        "word_log_freq_min": word_log_freq_min,
        "word_log_freq_max": word_log_freq_max,
    }
