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
    ENRICHED_WORDLIST_PATH,
    BIGRAM_PATH,
    TRIGRAM_PATH,
    WORD_FEATURES_PATH,
    SNIPPETS_PATH,
)

from app.ml.difficulty_features import (
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
    data = json.loads(ENRICHED_WORDLIST_PATH.read_text())

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
    bigrams = json.loads(BIGRAM_PATH.read_text())
    trigrams = json.loads(TRIGRAM_PATH.read_text())
    return bigrams, trigrams


# ------------------------------------------------------------
# Compute and store features for each individual word
# ------------------------------------------------------------
def generate_word_feature_vectors(words):
    feature_map = {}

    for w in words:
        feature_map[w] = compute_difficulty_features(w)

    WORD_FEATURES_PATH.write_text(json.dumps(feature_map, indent=2))
    return feature_map


# ------------------------------------------------------------
# Generate random multi-word snippets and compute difficulty
# ------------------------------------------------------------
def generate_snippets(words, n=20000, min_len=5, max_len=8):
    snippets = []

    for _ in range(n):
        k = random.randint(min_len, max_len)
        ws = random.sample(words, k)
        text = " ".join(ws)

        feats = compute_difficulty_features(text)

        snippets.append({
            "words": ws,
            "text": text,
            "features": feats,
        })

    SNIPPETS_PATH.write_text(json.dumps(snippets, indent=2))
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
    print(f"   → Word features saved to: {WORD_FEATURES_PATH}")
    print(f"   → Snippets saved to:      {SNIPPETS_PATH}")


if __name__ == "__main__":
    run()
