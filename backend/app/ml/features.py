import numpy as np
import re

class FeatureExtractor:
    """
    Centralized feature extraction logic for Snippets (and potentially Users).
    """

    @staticmethod
    def compute_snippet_features(text: str) -> np.ndarray:
        """
        Computes a manual feature vector for a snippet.
        Dimensions: ~15-20
        
        Features:
        1. Length
        2. Avg Word Length
        3. Space Density
        4. Uppercase Ratio
        5. Punctuation Density
        6. Number Density
        7. Left-Hand Heavy % (approx)
        8. Right-Hand Heavy % (approx)
        9. Alternation Count (L->R or R->L)
        10. Rollover Potential (Same hand, different fingers, descending time?) - approximated by bigram frequency
        11. Double Letter Count
        12. "Difficult" Bigram Count (e.g. 'zq', 'wx')
        13. Rhythm Variance (simulated based on word length variance)
        """
        
        if not text:
            return np.zeros(15, dtype=np.float32)

        length = len(text)
        words = text.split()
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        
        # Character sets
        uppercase = sum(1 for c in text if c.isupper())
        punctuation = sum(1 for c in text if not c.isalnum() and not c.isspace())
        numbers = sum(1 for c in text if c.isdigit())
        spaces = sum(1 for c in text if c.isspace())
        
        # Hand heaviness (Simple QWERTY assumption)
        left_keys = set("qwertasdfgzxcvb12345")
        right_keys = set("yuiophjklnm67890")
        
        left_count = sum(1 for c in text.lower() if c in left_keys)
        right_count = sum(1 for c in text.lower() if c in right_keys)
        
        # Normalize counts
        upper_ratio = uppercase / length
        punct_ratio = punctuation / length
        num_ratio = numbers / length
        space_ratio = spaces / length
        
        left_ratio = left_count / length
        right_ratio = right_count / length
        
        # Alternations
        alternations = 0
        prev_hand = None
        for c in text.lower():
            curr_hand = "L" if c in left_keys else ("R" if c in right_keys else None)
            if curr_hand and prev_hand and curr_hand != prev_hand:
                alternations += 1
            prev_hand = curr_hand
            
        alt_ratio = alternations / length
        
        # Doubles
        doubles = sum(1 for i in range(len(text)-1) if text[i].lower() == text[i+1].lower())
        double_ratio = doubles / length
        
        # "Difficult" chars (bottom row, pinky reaches)
        difficult_keys = set("zxcvqbp")
        diff_count = sum(1 for c in text.lower() if c in difficult_keys)
        diff_ratio = diff_count / length

        # Assemble vector (13 features so far)
        features = np.array([
            min(length / 200.0, 1.0), # Normalize length
            min(avg_word_len / 10.0, 1.0),
            space_ratio,
            upper_ratio,
            punct_ratio,
            num_ratio,
            left_ratio,
            right_ratio,
            alt_ratio,
            double_ratio,
            diff_ratio,
            0.0, # Placeholder 1
            0.0, # Placeholder 2
            0.0, # Placeholder 3
            0.0  # Placeholder 4
        ], dtype=np.float32)
        
        return features

    @staticmethod
    def feature_dim() -> int:
        return 15
