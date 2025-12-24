from app.config import settings

# S3 object keys within the offline data bucket
WORDLIST_KEY = "english_10k.json"
ENRICHED_WORDLIST_KEY = "english_10k_enriched.json"
BIGRAM_KEY = "bigram_freqs.json"
TRIGRAM_KEY = "trigram_freqs.json"
WORD_FEATURES_KEY = "word_features.json"
SNIPPETS_KEY = "snippets.json"

# Offline bucket name
OFFLINE_BUCKET = settings.data_bucket_offline
