# **FlowType Architecture Overview (v0.3.0 MVP)**

This document describes the system architecture, ML pipeline, and key design decisions behind FlowType's adaptive typing practice engine.

---

## **System Overview**

FlowType is a full-stack typing practice application that adapts snippet difficulty based on user skill. The system comprises:

1. **Frontend**: React + TypeScript + Vite (real-time UI, keystroke capture)
2. **Backend**: FastAPI + SQLAlchemy (API, user management, session logging)
3. **ML Pipeline**: Feature extraction + FAISS vector indexing + LinTS contextual bandit
4. **Database**: SQLite (dev) / PostgreSQL (prod)

---

## **Data Pipeline**

### **1. Snippet Dataset**

**Source**: 10,000 English words (curated wordlist)

**Enrichment**:
- **Zipf Frequencies**: Using wordfreq library for corpus-based word popularity
- **Bigram/Trigram Frequencies**: Co-occurrence statistics from Brown corpus
- **Ergonomic Metadata**: Key-distance, hand-switching patterns

**Snippet Generation**:
- Random word selection with frequency weighting
- Snippet length: 30–120 characters
- Variety via Markov chain (word transitions based on n-gram probabilities)
- ~3,000 unique snippets in database

### **2. Difficulty Feature Extraction**

Each snippet is characterized by **50+ engineered features**:

| Category | Features | Count |
|---|---|---|
| **Linguistic** | Avg word Zipf score, vocabulary diversity, word length distribution, sentence structure | 8 |
| **Character Distribution** | Char entropy, consonant/vowel ratio, symbol/punctuation density | 5 |
| **Ergonomic Load** | Shift-key %, number/punctuation %, bigram hand-switching, row-jumps, key-distance | 12 |
| **Sequence Patterns** | Consecutive same-hand pairs, finger stretches, alternation patterns | 8 |
| **Linguistic Complexity** | Phonetic features, compound words, syllable count, syllable patterns | 7 |
| **Positional Variance** | Early/mid/late character patterns, word-start/word-end difficulty | 5 |

**Normalization**: Z-score normalization (mean=0, std=1) within corpus.

### **3. Vectorization & Indexing**

- **Embedding**: 30-dimensional fixed-size vector (subset of normalized features)
- **Index**: FAISS flat index (L2 distance metric)
- **Lookup Speed**: ~10ms for top-100 neighbors

---

## **User Modeling**

### **Keystroke Telemetry**

Each keystroke logged with:
- **Timestamp** (millisecond precision)
- **Key** (character pressed)
- **Correctness** (match vs. target snippet)
- **Inter-keystroke Interval (IKI)** (time since previous keystroke)

### **Session Metrics**

Per-session statistics computed:
- **WPM**: (Characters typed / 5) / (Time in minutes)
- **Accuracy**: (Correct keystrokes / Total keystrokes) × 100%
- **Speed Distribution**: IKI percentiles (median, p95)
- **Backspace Rate**: Corrections made / Total characters
- **Session Duration**: Time to completion or timeout

### **User Profile (Rolling Stats)**

After each session, user stats updated with EMA (exponential moving average):
- **EMA WPM**: Weighted rolling average of last 10 sessions
- **EMA Accuracy**: Weighted rolling average of last 10 sessions
- **Consistency**: Variance of WPM over recent sessions
- **Best WPM by Mode**: Per-mode personal bests (15s, 30s, 60s, 120s)

### **Cold Start**

For new users without session history:
- Initialize with **median difficulty** snippet (50th percentile)
- Adapt after 2–3 sessions based on performance

---

## **Adaptive Difficulty Selection**

### **Algorithm: LinTS (Linear Thompson Sampling)**

**Problem**: Balance difficulty to optimize learning (not too easy, not too hard).

**Approach**:
1. **User Skill**: Represented as EMA WPM + accuracy
2. **Snippet Difficulty**: 30-dim feature vector (FAISS embedding)
3. **Candidate Retrieval**: FAISS top-100 nearest neighbors to user skill profile
4. **Bandit Policy**: Thompson sampling over difficulty posterior
5. **Reward Signal**: ΔWPM + β × ΔAccuracy (improvement vs. session average)

**Process**:
```
For each new session:
  1. Get user's last 5-session stats (EMA WPM, accuracy)
  2. Encode user as vector in feature space
  3. Query FAISS: top-100 snippets similar to user skill
  4. LinTS policy: sample difficulty target from posterior
  5. Rank candidates by distance to target → pick best match
  6. Log session: [user, snippet, WPM, accuracy, keystroke log]
  7. Update bandit posterior offline (nightly batch)
```

### **Safety Constraints**

- **Maximum difficulty jump**: Limited +10% per session (fatigue prevention)
- **Minimum difficulty floor**: Median snippet for struggling users
- **Free mode exception**: Best WPM not tracked (exploration encouraged)

---

## **Frontend Architecture**

### **Key Components**

**App.tsx** (Root)
- Manages session state (mode, paused, AFK)
- Routes between pages (Type, Stats, Leaderboard, Wiki)
- Renders AFK overlay on inactivity (5s idle timeout)

**TypingZone.tsx** (Main Typing Interface)
- Real-time keystroke capture
- Cursor positioning (char-by-char)
- Progress visualization
- AFK detection (5s keystroke inactivity)

**StatsPage.tsx**
- Lifetime stats (total sessions, total time, best WPM)
- Session history (last 20 sessions)
- Keyboard heatmap (Recharts-based)
- Activity heatmap (per-day session tracking)

**LeaderboardPage.tsx**
- Per-mode rankings (15s, 30s, 60s, 120s)
- Username + best WPM + session count
- Progression tips (accuracy → consistency → speed)

**Account Management Pages**
- **ChangeUsernamePage**: Update username
- **ChangePasswordPage**: Change password (verify current password)
- **DeleteAccountPage**: Two-step confirmation (password + typed confirmation)

### **Context & State Management**

**SessionModeContext**: Shared session state
- Current mode (15/30/60/120/free)
- Session started flag
- Pause flag
- Default mode: 15s

**AuthContext**: User authentication
- Username, user ID
- Auth token (JWT)
- Login/logout/register functions

---

## **Backend API**

### **Authentication**
- `POST /auth/register`: Create account
- `POST /auth/login`: Get JWT token
- `PUT /auth/users/change-username`: Update username
- `PUT /auth/users/change-password`: Update password (verify current)
- `DELETE /auth/users/delete-account`: Delete account + cascade delete sessions

### **Snippets**
- `GET /snippets/next?mode={mode}`: Retrieve next adaptive snippet

### **Sessions**
- `POST /sessions`: Log session (keystroke log, WPM, accuracy, mode)
- `GET /sessions/user`: Get user's session history

### **Leaderboards**
- `GET /leaderboards/{mode}`: Top-100 users by best WPM in mode

### **Users**
- `GET /users/{user_id}/stats`: Get user stats (lifetime, per-mode)

---

## **Database Schema (SQLite / PostgreSQL)**

### **Core Tables**

```python
# User
- id (INTEGER PRIMARY KEY)
- username (VARCHAR UNIQUE)
- hashed_password (VARCHAR)
- created_at (BIGINT)
- stats (JSON): {"lifetime_sessions": 10, "lifetime_wpm": 65.2, "consistency": 3.1}
- best_wpm_15, best_wpm_30, best_wpm_60, best_wpm_120 (FLOAT)

# Snippet
- id (INTEGER PRIMARY KEY)
- text (VARCHAR)
- length (INTEGER)
- difficulty_features (JSON): {50+ feature values}
- embedding (BLOB): 30-dim normalized vector
- created_at (BIGINT)

# Session
- id (INTEGER PRIMARY KEY)
- user_id (FK → User)
- snippet_id (FK → Snippet)
- mode (VARCHAR): "15", "30", "60", "120", "free"
- wpm (FLOAT)
- accuracy (FLOAT)
- keystroke_log (JSON): [{"key": "a", "ts": 1234, "correct": true}, ...]
- created_at (BIGINT)

# BanditState (RL parameters)
- id (INTEGER PRIMARY KEY)
- user_id (FK → User, UNIQUE)
- skill_posterior_mean (FLOAT)
- skill_posterior_cov (FLOAT)
- session_history (JSON): [{"reward": 5.2, "difficulty": 0.6}, ...]
- updated_at (BIGINT)
```

---

## **Performance Characteristics**

| Component | Latency | Bottleneck |
|---|---|---|
| **FAISS Retrieval** | ~10ms | Vector similarity search |
| **Bandit Policy** | ~5ms | Numpy/scipy ops |
| **API E2E** | <100ms | DB query + FAISS + policy |
| **Frontend Render** | 16ms (60fps) | React diffing + DOM updates |
| **Keystroke Capture** | <2ms | Native event listeners |

---

## **ML Loop**

### **Offline (Nightly Batch)**

```
For each user with recent sessions:
  1. Extract features from keystroke logs (WPM, accuracy, patterns)
  2. Update EMA stats
  3. Run Bayesian bandit posterior update
  4. Save updated skill posterior to BanditState
```

### **Online (Per-Session)**

```
At session start:
  1. Load user's skill posterior mean
  2. Query FAISS for top-K similar snippets
  3. Apply LinTS policy → select difficulty target
  4. Rank candidates → pick best match

At session end:
  1. Compute WPM, accuracy, keystroke metrics
  2. Calculate reward = ΔWPM + β·ΔAccuracy
  3. Queue for offline bandit update
  4. Update leaderboard cache
```

---

## **Testing & Monitoring**

### **Manual Testing**
- Real-time typing test (5–10 sessions per mode)
- Account management (username/password changes)
- AFK detection (5s idle trigger)
- Leaderboard consistency

### **Automated Testing (TODO)**
- Unit tests for API endpoints
- Integration tests for ML pipeline
- E2E tests for critical user flows
- Load testing for FAISS retrieval

### **Logging & Monitoring**
- Application logs: uvicorn + custom loggers
- Session logs: JSON keystroke archives
- Performance metrics: API latency, FAISS query time
- User metrics: WPM distribution, leaderboard churn

---

## **Technology Stack**

| Layer | Technology |
|---|---|
| **Frontend** | React 18 + TypeScript + Vite |
| **UI Library** | Tailwind CSS + Recharts + Lucide Icons |
| **Backend** | FastAPI + Uvicorn |
| **ORM** | SQLAlchemy |
| **Database** | SQLite (dev) / PostgreSQL (prod) |
| **ML** | NumPy + SciPy + FAISS + scikit-learn |
| **ML Models** | GRU (user profiling) + LinTS (bandit) |
| **Migrations** | Alembic |
| **Auth** | JWT (PyJWT) + bcrypt |
| **Containerization** | Docker + Docker Compose |

---

## **Future Enhancements**

### **ML Improvements**
- Hierarchical bandits (coarse → fine granularity)
- Transfer learning for cold-start users
- Feature importance analysis (SHAP)
- Real-time online learning (not batched)

### **UX Enhancements**
- Difficulty labels ("Warmup" vs. "Challenge")
- Personalized practice recommendations
- Keystroke-level feedback ("left pinky errors")
- Skill progression trends (daily/weekly)

### **Platform Expansion**
- Mobile app (React Native)
- Multiplayer modes (competitive)
- Custom wordlists
- API for third-party integrations
