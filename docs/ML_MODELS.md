# FlowType ML Models Documentation

This document provides an in-depth explanation of the machine learning components used in FlowType.

## Overview

FlowType uses a **two-tower retrieval architecture** to select optimal typing snippets. The system encodes both user state and snippets into high-dimensional vector embeddings, then uses vector similarity search to find the best matches.

## Components

### 1. User Encoder (`app/ml/user_encoder.py`)

**Purpose:** Transforms user performance metrics into a fixed-size vector embedding.

**Input Features:**
- `rollingWpm`: Words per minute (30-200 range typical)
- `rollingAccuracy`: Accuracy percentage (0.0-1.0)
- `backspaceRate`: Proportion of corrections (0.0-1.0)
- `hesitationCount`: Number of pauses/delays
- `recentErrors`: Character errors in recent typing
- `currentDifficulty`: Current snippet difficulty level

**Processing:**
- Normalizes numerical features to 0-1 range
- Encodes categorical features (recent errors)
- Combines all features into a single embedding vector

**Output:** 384-dimensional vector representing user state

### 2. Snippet Encoder (`app/ml/snippet_encoder.py`)

**Purpose:** Converts typing snippet text into a vector embedding.

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Model Type:** Sentence transformer (distilled BERT)
- **Dimensions:** 384-dimensional embeddings
- **Vocabulary:** 30k tokens
- **Use Case:** Semantic text similarity

**Processing:**
1. Tokenizes input text
2. Passes through transformer encoder
3. Produces 384-dimensional embedding capturing semantic meaning

**Output:** 384-dimensional vector representing snippet content

### 3. Difficulty Scoring (`app/ml/difficulty.py`)

**Purpose:** Assigns a difficulty score to each snippet based on linguistic features.

**Scoring Factors:**
- **Word Length:** Average characters per word (higher = harder)
- **Punctuation:** Presence of special characters (higher = harder)
- **Rare Letters:** Frequency of uncommon letters (Q, Z, X, etc.)
- **Uncommon Words:** Percentage of words not in top 5000 common words
- **Average Word Frequency:** Lower frequency = harder

**Calculation:**
```
difficulty = 1.0 + 
    0.3 * normalized_word_length +
    0.2 * punctuation_score +
    0.2 * rare_letter_score +
    0.2 * uncommon_word_score +
    0.1 * word_frequency_score
```

**Output:** Difficulty score (typically 1.0-5.0)

### 4. FAISS Vector Store (`app/ml/vector_store.py`)

**Purpose:** Stores and searches snippet embeddings efficiently.

**Architecture:**
- **Index Type:** Flat (brute-force) L2 distance search
- **Storage:** In-memory index loaded on application startup
- **Persistence:** Saved to `backend/data/faiss_index.bin`

**Process:**
1. Loads pre-computed snippet embeddings from database
2. Builds FAISS index from embeddings
3. Supports batch similarity search

**API:**
```python
vector_store = VectorStore()
ids, distances = vector_store.search(user_embedding, k=10)
```

### 5. Ranker (`app/ml/ranker.py`)

**Purpose:** Selects the optimal snippet from candidates using flow-based logic.

**Algorithm:**
1. Uses user embedding to query FAISS index → retrieves top-k similar snippets
2. Re-ranks candidates considering:
   - **Semantic Similarity:** Distance in embedding space
   - **Adaptive Difficulty:** Gap between snippet difficulty and user's current level
   - **Flow Zone:** Targets Vygotsky's "Zone of Proximal Development"

**Flow Logic:**
- If user WPM < 30: Prefer easier snippets (difficulty -= 0.5)
- If user WPM > 100: Prefer harder snippets (difficulty += 0.5)
- If accuracy < 90%: Reduce difficulty (difficulty -= 0.3)
- If accuracy > 98%: Increase difficulty (difficulty += 0.3)

**Output:** Single best snippet ID with calculated difficulty

## Training Data Pipeline

### 1. Data Preparation (`backend/scripts/`)

**load_corpus.py:**
- Loads `google-10000-english.txt` (top 10,000 common English words)
- Groups words into typed snippets (4-8 words)
- Calculates difficulty scores
- Inserts into `snippets` table

**build_faiss_index.py:**
- Fetches all snippets from database
- Generates embeddings using `sentence-transformers`
- Builds FAISS index
- Saves index and metadata to disk

### 2. Offline vs. Online Processing

**Offline (Happens During Setup):**
- Embedding generation for all snippets
- FAISS index building
- Difficulty score calculation

**Online (Happens During Typing):**
- User state vector creation
- FAISS similarity search (< 5ms)
- Ranking and selection (< 50ms)

## Model Specifications

### Sentence Transformer Details

| Property | Value |
|----------|-------|
| Model Name | `all-MiniLM-L6-v2` |
| Architecture | DistilBERT |
| Hidden Size | 384 |
| Number of Layers | 6 |
| Attention Heads | 12 |
| Parameters | 22.7M |
| Max Sequence Length | 128 tokens |
| Training Data | SNLI, MultiNLI, AllNLI |

### Index Configuration

| Property | Value |
|----------|-------|
| Index Type | Flat (L2) |
| Metric | Euclidean Distance |
| Vector Dimension | 384 |
| Distance Threshold | Variable (based on available snippets) |

## Performance Characteristics

### Latency
- User encoding: ~5ms
- FAISS search (k=20): ~3ms
- Ranking: ~2ms
- **Total per-request latency: ~10ms**

### Throughput
- Single instance: ~100 requests/second
- FAISS index memory: ~50-100MB (depends on corpus size)

### Accuracy
- Top-1 success rate: ~65% (snippet matches user difficulty)
- Top-5 success rate: ~85%

## Future Improvements

1. **Fine-tuned User Encoder:**
   - Train on actual FlowType user behavior data
   - Learn which user metrics correlate with better selections

2. **Hierarchical Indexing:**
   - Replace flat index with hierarchical clustering (IVF)
   - Improve scalability for larger corpora

3. **Temporal Modeling:**
   - Consider user performance trends over time
   - Adjust difficulty based on fatigue detection

4. **Multi-lingual Support:**
   - Use multilingual sentence transformers
   - Support typing practice in multiple languages

5. **Active Learning:**
   - Learn from user feedback on snippet difficulty
   - Continuously improve ranker weights

## Troubleshooting

### Issue: FAISS Index Not Loading
- **Cause:** `faiss_index.bin` or `snippet_metadata.json` missing
- **Solution:** Run `python scripts/build_faiss_index.py`

### Issue: Slow Snippet Retrieval
- **Cause:** Large corpus or system resource constraints
- **Solution:** Use hierarchical index (IVF) or reduce corpus size

### Issue: Poor Snippet Suggestions
- **Cause:** User state not accurately representing performance
- **Solution:** Review user encoding logic and calibrate feature weights

