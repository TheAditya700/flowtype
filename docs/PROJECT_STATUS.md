# **FlowType Project Status & Roadmap (v0.3.0 - MVP Release)**

This document summarizes the current status, completed features, and roadmap of **FlowType**, an adaptive typing practice application.

---

# **Project Status**

**Current Version:** v0.3.0 — MVP Release  
**Status:** Core features complete and shipped; ready for user testing

---

# ✅ **Completed Features**

## **Backend Core**
- [x] FastAPI application with modular router structure
- [x] SQLite/PostgreSQL support with SQLAlchemy ORM
- [x] Alembic database migrations
- [x] JWT authentication with password hashing
- [x] User account management (register, change username, change password, delete account)

## **Snippet System**
- [x] Enriched 10k wordlist with frequency data
- [x] Snippet generation with 4 timed modes (15s, 30s, 60s, 120s) + free mode
- [x] 50+ difficulty features (linguistic & ergonomic)
- [x] Feature normalization (z-score)
- [x] FAISS vector index for fast retrieval
- [x] Adaptive difficulty selection based on user skill

## **ML Pipeline**
- [x] User feature extraction (keystroke telemetry, rolling stats)
- [x] GRU-based user profile from session history
- [x] EMA (exponential moving average) stats tracking
- [x] Per-character accuracy & speed analysis
- [x] LinTS (contextual bandit) for difficulty selection
- [x] Reward calculation (WPM improvement, consistency)
- [x] Session-based feature aggregation

## **Session Tracking**
- [x] Real-time keystroke capture (timestamp, key, correctness)
- [x] Per-snippet performance logging
- [x] Rolling WPM & accuracy calculation
- [x] Session completion and archival
- [x] Best WPM tracking (timed modes only; free mode excluded)
- [x] Partial snippet handling (for timed mode timeout)

## **Frontend - Core Pages**
- [x] **Type Page**: Real-time typing interface, mode selector, cursor positioning
- [x] **Stats Page**: Lifetime stats, session history, keyboard heatmap, progress charts
- [x] **Leaderboard Page**: All-time rankings by mode (15s, 30s, 60s, 120s)
- [x] **Wiki Page**: Reference documentation
- [x] **Auth Page**: Login/register modal
- [x] **Account Management Pages**: Change username, change password, delete account (with confirmation)

## **Frontend - Features**
- [x] Session mode context (15/30/60/120/free)
- [x] Header with navigation, mode selector, user menu
- [x] Header visibility control (hide while typing, show when paused)
- [x] Mode selector disabling during active session
- [x] Enter-to-pause (timed modes only)
- [x] AFK detection (5s inactivity timeout with restart overlay)
- [x] Live stats display (WPM, accuracy, time remaining)
- [x] Results dashboard with session replay
- [x] Keyboard heatmap (Recharts-based)
- [x] Activity heatmap (per-day session tracking)
- [x] Responsive design (mobile-friendly)

## **UI/UX**
- [x] Dark theme with consistent styling
- [x] Tailwind CSS + custom color scheme
- [x] Icon integration (lucide-react)
- [x] Smooth transitions and animations
- [x] Error handling and user feedback
- [x] Loading states and placeholders
- [x] Modal dialogs and confirmations

## **DevOps & Tooling**
- [x] Docker support (Dockerfile for backend & frontend)
- [x] Docker Compose for local development
- [x] Environment configuration via .env
- [x] Database initialization scripts
- [x] FAISS index building scripts
- [x] Data cleanup & maintenance scripts

## **Code Quality**
- [x] Type safety (TypeScript frontend, Python typing backend)
- [x] Removed dead code (unused components, utilities)
- [x] Removed legacy artifacts (two-tower phase pycache)
- [x] Clean module imports and exports
- [x] Consistent naming conventions

---

# 🚧 **In Progress / Partial**

## **Analytics**
- [ ] Advanced heatmaps (bigram frequency analysis)
- [ ] Skill progression charts (daily/weekly trends)
- [ ] Weakness identification by key clusters
- [ ] Practice recommendations based on weak keys

## **RL Loop**
- [ ] Hyperparameter tuning for LinTS
- [ ] Reward shaping refinements
- [ ] Cold-start improvements
- [ ] Session feedback loop integration

---

# ❌ **Not Planned for MVP**

- Custom wordlists / user-generated content
- Multiplayer / competitive modes
- Mobile app (native)
- Real-time collaboration
- Advanced ML features (transformer-based ranking)
- Telemetry dashboard (backend-only)
- Social features (teams, challenges, streaming)

---

# **Architecture Highlights**

## **Adaptive Difficulty**
- User profile built from keystroke telemetry
- Difficulty estimated via 50+ engineered features
- LinTS bandit balances exploration vs. exploitation
- FAISS retrieves top-K candidates, bandit picks best

## **Persistence**
- Session data archived with full keystroke logs
- User stats computed offline from session history
- Leaderboard updated on session completion
- No real-time rank updates (for performance)

## **Performance**
- FAISS index: ~10ms retrieval (top-100 candidates)
- API response: <100ms (including DB + vector lookup)
- Frontend: Vite dev server with HMR
- Database: Indexed on user_id, snippet_id, created_at

---

# **Known Limitations**

1. **Free Mode**: Best WPM not tracked (intentional—focus on learning, not scores)
2. **AFK Timeout**: 5 seconds of inactivity triggers restart (no recovery)
3. **Leaderboard**: All-time only; no time-windowed rankings (yet)
4. **Heatmap**: Per-character accuracy; bigram analysis is comment-only
5. **Stats**: Limited to 6 months of history (configurable)

---

# **Future Roadmap (Post-MVP)**

### **Phase 2: Analytics & UX**
- Time-windowed leaderboards (this week, this month)
- Advanced keyboard heatmaps (bigram frequency)
- Skill progression visualization
- Daily practice streaks

### **Phase 3: Advanced RL**
- Hierarchical bandit (coarse-grained → fine-grained)
- Fatigue detection (performance decay over time)
- Personalized difficulty curve
- Session recommendation engine

### **Phase 4: Social & Community**
- User profiles with public stats
- Friend leaderboards
- Challenge creation & sharing
- Team practice sessions

### **Phase 5: Ecosystem**
- Mobile app (React Native / Flutter)
- Integrations (Discord, Twitch)
- API for third-party apps
- Content marketplace (custom wordlists)

---

# **Tech Debt & Cleanup Done**

- ✅ Removed unused utility files (`metrics.py`, `preprocessing.py`)
- ✅ Cleaned up pycache artifacts from two-tower phase
- ✅ Removed 10 unused React components
- ✅ Updated comments (removed two-tower references)
- ✅ Consolidated imports and exports
- ✅ Verified all active code paths

---

# **Testing & QA**

**Current:** Manual testing (no automated test suite yet)  
**To-Do:**
- Unit tests for API endpoints
- Integration tests for ML pipeline
- E2E tests for critical user flows
- Load testing for FAISS retrieval

---

# **Deployment**

**Development:** `npm run dev` (frontend) + `uvicorn` (backend)  
**Production:** Docker Compose + PostgreSQL + gunicorn (planned)  
**Hosting:** Ready for Vercel (frontend) + Railway/Render (backend)

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
