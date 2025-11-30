from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

WORDLIST_PATH = DATA_DIR / "english_10k.json"
ENRICHED_WORDLIST_PATH = DATA_DIR / "english_10k_enriched.json"
BIGRAM_PATH = DATA_DIR / "bigram_freqs.json"
TRIGRAM_PATH = DATA_DIR / "trigram_freqs.json"
WORD_FEATURES_PATH = DATA_DIR / "word_features.json"
SNIPPETS_PATH = DATA_DIR / "snippets.json"
