# generator/generate.py
# ------------------------------------------------------------
# FINAL VERSION — ENRICHED + N-GRAM-INTEGRATED
# - Uses enriched Monkeytype english_10k_enriched.json
# - Loads weighted bigram/trigram tables from /data
# - Injects global ngram + wordfreq tables into difficulty_features
# - Generates word-level + snippet-level feature vectors
# ------------------------------------------------------------

import json
import random
from pathlib import Path

from .config import (
    ENRICHED_WORDLIST_KEY,
    BIGRAM_KEY,
    TRIGRAM_KEY,
    WORD_FEATURES_KEY,
    SNIPPETS_KEY,
    OFFLINE_BUCKET,
)
from app.utils.s3_data import read_json, write_json

from app.ml.snippet_features import (
    compute_difficulty_features,
    set_global_ngram_tables,
    set_global_wordfreq,
)


# ------------------------------------------------------------
# Load enriched wordlist
# Expected format:
# {
#   "name": "english_10k",
#   "words": [
#       {"word": "the", "zipf": 7.4},
#       {"word": "of",  "zipf": 7.1},
#       ...
#   ]
# }
# ------------------------------------------------------------
def load_enriched_wordlist():
    data = read_json(OFFLINE_BUCKET, ENRICHED_WORDLIST_KEY, env="offline")

    words = []
    word_freq = {}

    for entry in data["words"]:
        w = entry["word"].lower().strip()
        zipf = float(entry.get("zipf", 1.0))
        words.append(w)
        word_freq[w] = zipf

    return words, word_freq


# ------------------------------------------------------------
# Load n-gram frequency tables (already weighted)
# ------------------------------------------------------------
def load_ngram_tables():
    bigrams = read_json(OFFLINE_BUCKET, BIGRAM_KEY, env="offline")
    trigrams = read_json(OFFLINE_BUCKET, TRIGRAM_KEY, env="offline")
    return bigrams, trigrams


# ------------------------------------------------------------
# Compute and store features for each individual word
# ------------------------------------------------------------
def generate_word_feature_vectors(words):
    feature_map = {}

    for w in words:
        feature_map[w] = compute_difficulty_features(w)

    write_json(OFFLINE_BUCKET, WORD_FEATURES_KEY, feature_map, env="offline")
    return feature_map


# ------------------------------------------------------------
# Generate random multi-word snippets and compute difficulty
# ------------------------------------------------------------
def generate_snippets(words, n=20000, min_len=5, max_len=8):
    snippets = []

    attempts = 0
    max_attempts = n * 5  # Prevent infinite loops

    while len(snippets) < n and attempts < max_attempts:
        attempts += 1
        k = random.randint(min_len, max_len)
        ws = random.sample(words, k)
        text = " ".join(ws)

        if len(text) > 60:
            continue

        feats = compute_difficulty_features(text)

        snippets.append(
            {
                "words": ws,
                "text": text,
                "features": feats,
            }
        )

    write_json(OFFLINE_BUCKET, SNIPPETS_KEY, snippets, env="offline")
    return snippets


# ------------------------------------------------------------
# Main runner
# ------------------------------------------------------------
def run():
    print("==> Loading enriched wordlist…")
    words, word_freq = load_enriched_wordlist()

    print("==> Loading n-gram tables…")
    bigram_freqs, trigram_freqs = load_ngram_tables()

    print("==> Injecting global frequency tables…")
    set_global_ngram_tables(bigram_freqs, trigram_freqs)
    set_global_wordfreq(word_freq)

    print("==> Computing per-word difficulty features…")
    generate_word_feature_vectors(words)

    print("==> Generating snippet difficulty features…")
    generate_snippets(words)

    print("\n✔ Generation complete!")
    print(f"   → Word features saved to: s3://{OFFLINE_BUCKET}/{WORD_FEATURES_KEY}")
    print(f"   → Snippets saved to:      s3://{OFFLINE_BUCKET}/{SNIPPETS_KEY}")


if __name__ == "__main__":
    run()
