# **FlowType Project Status & Roadmap (Updated – 2025)**

This document summarizes the current architecture, ML pipeline, backend/frontend progress, and future roadmap of the **FlowType adaptive typing engine**.

---

# **Project Status**

## **Current Version**

**v0.2.0 — Engine Rewrite Milestone**

## **Completion Status**

---

# ✅ **Completed Features**

## **Core Backend**

* [x] FastAPI application + modular router structure
* [x] SQLite/PostgreSQL support via SQLAlchemy
* [x] New **snippet schema** with:

  * engineered difficulty feature vectors
  * normalized feature vectors
  * stored embedding vectors
* [x] Snippet generation pipeline
* [x] Enriched 10k wordlist with wordfreq Zipf values
* [x] Weighted bigram/trigram frequency generation
* [x] Difficulty feature extractor (50+ ergonomic & linguistic features)
* [x] Snippet vectorization + normalization
* [x] FAISS index integration (top-K candidate retrieval)
* [x] Basic snippet ranking service
* [x] Keystroke telemetry ingestion
* [x] RQ background workers + Redis task queue

## **ML Pipeline**

* [x] Word enrichment (Zipf frequencies)
* [x] Weighted n-gram generation
* [x] Snippet generation with synthetic variety
* [x] Full difficulty feature computation for every snippet
* [x] Vectorized snippet embeddings (fixed 30-dim engineered vectors)
* [x] Normalization pipeline (z-score or min-max)
* [x] FAISS index builder
* [x] Per-word difficulty vector store
* [x] Overall two-tower architecture scaffolding
* [x] Telemetry logging (raw keystrokes)

## **Frontend Core**

* [x] React + Vite + TypeScript
* [x] Real-time keystroke capture
* [x] WPM, accuracy, backspace rate, hesitation spikes
* [x] Rolling difficulty estimation
* [x] Session completion & results
* [x] Beautiful Tailwind UI
* [x] Integrated snippet retrieval API

## **DevOps / Tools**

* [x] Docker containerization
* [x] Clean repo structure
* [x] Alembic migrations
* [x] SQLite local DB browsing workflow
* [x] Project-wide type safety (TS + Python typing)

---

# 🚧 **In Progress / Partially Done**

## **RL / Bandits Layer**

* [ ] Contextual bandit with UCB / Thompson
* [ ] Top-K FAISS candidate → bandit policy
* [ ] Reward shaping for improvement (ΔWPM, ΔAccuracy)
* [ ] Safety caps for fatigue

## **Backend Enhancements**

* [ ] Snippet difficulty fine-tuning
* [ ] Caching around snippet retrieval
* [ ] Performance tests & DB indexing

## **Frontend Enhancements**

* [ ] Mobile-first UI
* [ ] Keyboard heatmap visualization
* [ ] Weak-sequence highlighting

---

# ⏳ **Not Started**

* Leaderboards
* Social/club mode
* Achievements & progression
* Multi-language support
* Keyboard layout optimizer
* iOS/Android app
* Offline PWA mode
* Real-time multiplayer
* Personal training plans

---

# **Key Metrics & Benchmarks (Updated)**

| Metric                     | Target         | Current        |
| -------------------------- | -------------- | -------------- |
| API response               | < 40ms         | ~10–15ms       |
| FAISS lookup               | < 5ms          | 2–3ms          |
| Snippet generation         | n/a            | 20k in ~4s     |
| Difficulty feature compute | <0.5ms/snippet | ~0.3ms/snippet |
| Snippet vector norm        | <1ms           | ~0.5ms         |
| Typing latency             | < 50ms         | 15–25ms        |

---

# **Architecture Overview (Updated)**

```
┌──────────────────────────────────────────────┐
│                React Frontend                │
│  - Real-time keystroke telemetry             │
│  - Rolling WPM/accuracy                      │
│  - Session state + difficulty HUD            │
└───────────────┬──────────────────────────────┘
                │ REST API
                ▼
┌──────────────────────────────────────────────┐
│                  FastAPI                     │
│  ┌────────────────────────────────────────┐  │
│  │ Snippet Retrieval Pipeline             │  │
│  │ 1. Build user state U                 │  │
│  │ 2. FAISS ANN → top-K candidates       │  │
│  │ 3. Ranking via two-tower network      │  │
│  │ 4. (Future) RL/Bandit selection       │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │ ML Engine                              │  │
│  │ - Difficulty feature computation       │  │
│  │ - N-gram scoring                       │  │
│  │ - Snippet encoder                      │  │
│  │ - User GRU encoder (planned)           │  │
│  │ - RL Bandit Agent (planned)            │  │
│  └────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────┐  │
│  │ Data Layer                             │  │
│  │ - Snippets (text, features, vectors)   │  │
│  │ - Users                                │  │
│  │ - Sessions                             │  │
│  │ - Raw keystrokes                       │  │
│  └────────────────────────────────────────┘  │
└───────────────┬──────────────────────────────┘
                ▼
           SQLite / PostgreSQL
```

---

# **Data Model (Updated)**

## **Snippets**

Includes ML-ready vectors:

```sql
id                  UUID PK
text                TEXT
words               JSON
word_count          INT

features            JSON        -- raw 50-dim metrics
features_norm       JSON        -- normalized numeric vector
embedding           JSON        -- final 30-dim embedding

difficulty_score    FLOAT NULL  -- learned later
created_at          TIMESTAMP
```

## **Sessions**

```sql
id, user_id
started_at
duration_seconds
words_typed
errors
backspaces
final_wpm
avg_wpm
peak_wpm
accuracy
starting_difficulty
ending_difficulty
avg_difficulty
keystroke_events JSON
flow_score
```

## **Telemetry (Raw Keystrokes)**

Fully preserved for GRU training.

---

# **Tech Stack (Updated)**

### **Backend**

* FastAPI
* SQLAlchemy
* Alembic
* FAISS (ANN search)
* NumPy / SciPy
* Redis + RQ
* wordfreq
* Custom difficulty engine
* Python 3.11

### **Frontend**

* React
* TypeScript
* Vite
* Tailwind
* Zustand/Context (state)

### **ML**

* Engineered feature pipeline
* N-gram scoring
* Two-tower architecture
* RL bandits
* GRU keystroke encoder (planned)

---

# **Roadmap (Updated)**

## ⭐ **Phase 1 — Engine Rewrite (Completed)**

* Difficulty feature extractor
* N-gram weighted tables
* Full snippet generation pipeline
* Normalized embeddings
* Database rewrite + migrations
* FAISS indexing
* Snippet retrieval engine

## ⭐ **Phase 2 — Two-Tower Model (Core Implemented)**

**User Tower (GRU + structured stats):**

* [x] GRU keystroke encoder
* [x] Structured user stats encoder
* [ ] Fatigue modeling (Future)
* [ ] Burst detection (Future)

**Snippet Tower:**

* [x] 30-d difficulty vector
* [ ] Optional character CNN

**Joint Scoring:**

* [x] Bilinear head (User ⨂ Snippet)

## ⭐ **Phase 3 — RL + Curriculum Learning**

* Contextual bandit for snippet difficulty
* UCB / Thompson sampling
* Reward shaping
* FAISS top-K → Bandit policy
* Real-time adaptation

## ⭐ **Phase 4 — User Platform**

* Auth
* Persistent history
* Dashboards
* Weak area analytics
* Long-term progression

## ⭐ **Phase 5 — Community Layer**

* Leaderboards
* Clubs
* Social challenges
* Weekly events
* Multiplayer typing races

## ⭐ **Phase 6 — Expansion**

* Languages: EN → multi-language
* Keyboard layouts
* Mobile app
* Offline mode
* API for 3rd party platforms

---

# **Known Issues / Current Limitations**

* Snippet text is synthetic, not narrative
* No RL exploration yet (only ranking)
* GRU user tower not yet implemented
* No authentication
* Limited mobile experience
* No leaderboards or community layer

---

# **Success Metrics**

### Learning Metrics

* WPM improvement over sessions
* Accuracy stability
* Fatigue prediction quality
* RL reward growth

### System Metrics

* Latency < 30ms
* FAISS lookup < 3ms
* Memory < 300MB

---

# **Want me to generate a README version, or separate docs (e.g., `ML_ARCHITECTURE.md` and `SYSTEM_DESIGN.md`)?**
