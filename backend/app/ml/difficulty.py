import string
from collections import Counter

# Based on https://en.wikipedia.org/wiki/Letter_frequency
LETTER_FREQ = {
    'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 
    'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78, 
    'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97, 
    'P': 1.93, 'B': 1.29, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 
    'Q': 0.10, 'Z': 0.07
}
RARE_LETTERS = {letter for letter, freq in LETTER_FREQ.items() if freq < 1.0}

def calculate_difficulty(words: str) -> dict:
    """
    Calculates various difficulty metrics for a given string of words.
    Returns a dictionary of features.
    """
    words_list = words.split()
    word_count = len(words_list)
    if word_count == 0:
        return {
            "difficulty_score": 1.0, "avg_word_length": 0, "punctuation_density": 0,
            "rare_letter_count": 0, "bigram_rarity": 0
        }

    # 1. Average word length
    avg_word_length = sum(len(w) for w in words_list) / word_count

    # 2. Punctuation density
    punct_count = sum(1 for char in words if char in string.punctuation)
    punctuation_density = punct_count / len(words)

    # 3. Rare letter count
    upper_words = words.upper()
    rare_letter_count = sum(1 for char in upper_words if char in RARE_LETTERS)

    # 4. Bigram rarity (placeholder)
    # A real implementation would use a frequency map of common bigrams
    bigram_rarity = 0.0

    # Combine into a single score (weights are arbitrary and need tuning)
    difficulty_score = (
        (avg_word_length * 0.5) +
        (punctuation_density * 10) +
        (rare_letter_count * 0.2)
    )
    
    # Normalize to a 1-10 scale (very roughly)
    difficulty_score = min(max(difficulty_score, 1.0), 10.0)

    return {
        "difficulty_score": difficulty_score,
        "avg_word_length": avg_word_length,
        "punctuation_density": punctuation_density,
        "rare_letter_count": rare_letter_count,
        "bigram_rarity": bigram_rarity
    }
