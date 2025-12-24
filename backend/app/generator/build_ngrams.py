import json
import math
from collections import Counter

from . import config
from app.utils.s3_data import read_json, write_json

ENRICHED_WORDLIST_KEY = config.ENRICHED_WORDLIST_KEY
BIGRAM_KEY = config.BIGRAM_KEY
TRIGRAM_KEY = config.TRIGRAM_KEY
BUCKET = config.OFFLINE_BUCKET

# Fallbacks to match difficulty_features.py
UNKNOWN_BIGRAM = -15.0
UNKNOWN_TRIGRAM = -20.0


def load_enhanced_wordlist():
    """
    Loads enriched wordlist of shape:
    {
        "words": [
            { "word": "the", "zipf": 7.4 },
            ...
        ]
    }
    """
    data = read_json(BUCKET, ENRICHED_WORDLIST_KEY, env="offline")
    return data["words"]  # list of dicts


def log_scale(val: float, unknown_value: float) -> float:
    """
    Convert weighted counts into log10 frequency.
    Ensures values remain within reasonable human-scale ranges.
    """
    if val <= 0:
        return unknown_value

    return round(math.log10(val + 1.0), 6)


def build_weighted_ngrams():
    words = load_enhanced_wordlist()

    bigram_counter = Counter()
    trigram_counter = Counter()

    # ---------------------------------------------------------
    # Stage 1 — Weighted counting
    # ---------------------------------------------------------
    for item in words:
        w = item["word"].lower().strip()
        zipf = float(item.get("zipf", 1.0))

        # Guarantee positive weight
        if zipf <= 0:
            zipf = 1.0

        chars = list(w)
        L = len(chars)

        # Weighted bigrams
        for i in range(L - 1):
            bg = chars[i] + chars[i + 1]
            bigram_counter[bg] += zipf  # type: ignore

        # Weighted trigrams
        for i in range(L - 2):
            tg = chars[i] + chars[i + 1] + chars[i + 2]
            trigram_counter[tg] += zipf  # type: ignore

    # ---------------------------------------------------------
    # Stage 2 — Convert counters → scaled frequencies
    # ---------------------------------------------------------
    bigram_scores = {
        bg: log_scale(score, UNKNOWN_BIGRAM) for bg, score in bigram_counter.items()
    }

    trigram_scores = {
        tg: log_scale(score, UNKNOWN_TRIGRAM) for tg, score in trigram_counter.items()
    }

    # ---------------------------------------------------------
    # Stage 3 — Save files
    # ---------------------------------------------------------
    write_json(BUCKET, BIGRAM_KEY, bigram_scores, env="offline")
    write_json(BUCKET, TRIGRAM_KEY, trigram_scores, env="offline")

    print(f"Saved {len(bigram_scores)} log-scaled bigrams → s3://{BUCKET}/{BIGRAM_KEY}")
    print(
        f"Saved {len(trigram_scores)} log-scaled trigrams → s3://{BUCKET}/{TRIGRAM_KEY}"
    )


if __name__ == "__main__":
    build_weighted_ngrams()
