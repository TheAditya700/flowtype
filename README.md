# FlowType

Adaptive typing practice with a contextual bandit, FAISS retrieval, and keystroke-level telemetry across WPM, accuracy, smoothness, and rollover.

FlowType is a personal research + product project exploring adaptive difficulty and motor-skill learning using real keystroke telemetry.

The system is intentionally **interpretability-first**, drawing from research in human motor learning and typing dynamics rather than opaque end-to-end models. Adaptation is driven by explicit, inspectable signals—inter-key intervals, rollover, chunking, and error dynamics—that are surfaced directly to both the user and the learning algorithm.

This design choice follows prior work showing that fine-grained keystroke timing contains rich, stable structure for modeling skill, learning, and cognition.

## Research foundations

FlowType’s metric design and adaptation loop are grounded in established research on typing dynamics, motor control, and keystroke timing:

- **Yin et al. (CHI 2018)** — *“How Do We Type? Movement Strategies and Performance in Everyday Typing”*  
  https://userinterfaces.aalto.fi/136Mkeystrokes/resources/chi-18-analysis.pdf  
  Large-scale analysis of **136 million keystrokes** from everyday typing.  
  Demonstrates that expert performance emerges from **rhythmic timing, rollover behavior, chunked motor plans, and reduced variance**, motivating FlowType’s emphasis on IKIs, rollover, chunk length, and smoothness rather than WPM alone.

- **Logan & Crump (2011)** — *“Hierarchical control of cognitive processes: The case for skilled typewriting”*  
  https://www.sciencedirect.com/science/chapter/bookseries/abs/pii/B9780123855275000012  
  Shows that expert typing is governed by **hierarchical motor programs**, not character-level cognition, directly motivating FlowType’s chunking, fluency, rollover, and per-hand metrics.

- **Killourhy & Maxion (2009)** — *“Comparing anomaly-detection algorithms for keystroke dynamics”*  
  https://ieeexplore.ieee.org/document/5270346  
  Establishes that **inter-key interval distributions and variance** are stable, information-rich signals, supporting the use of IKI CV and spike-rate as core smoothness metrics.

Together, these works motivate FlowType’s focus on **interpretable timing-based signals and bounded, incremental adaptation**, rather than black-box sequence models.


## Headlines
- Metric-first typing surface: WPM, raw WPM, accuracy, smoothness (IKI CV + spike-rate), rollover, and per-hand fluency from every session.
- Contextual bandit (LinTS) steers snippet selection with a 16-dim embedding and a 130-dim user state (EMA + variance + previous snippet).
- Full keystroke telemetry feeds dashboards (speed series, replay events, heatmaps) and keeps the model reward grounded in user behavior.

## Demo / screens
<br>

![Typing Surface](screenshots/type.png)  
**Typing surface** — Adaptive snippet selection with real-time keystroke capture; every keydown/keyup feeds IKI, rollover, and chunking metrics used by the model.
<br>
<br>

![Session Results](screenshots/results.png)  
**Session results** — Post-session breakdown of WPM vs raw WPM, accuracy, smoothness, rollover, and flow metrics derived directly from keystroke timing.
<br>
<br>

![Stats Dashboard](screenshots/stats.png)  
**Stats dashboard** — Longitudinal view of speed, accuracy, and fluency trends with EMA smoothing, enabling users to see exactly how the model perceives their skill over time.
<br>
<br>

## What the system does
- Serves adaptive typing snippets via FastAPI + FAISS, guided by a LinTS contextual bandit.
- Collects keystroke-by-keystroke telemetry to compute WPM, raw WPM, accuracy, smoothness, rollover, and fluency per hand/cross-hand.
- Updates user feature EMAs after each session and persists them to Postgres for cold-start recovery.
- Returns rich session analytics (speed timeline, replay events, heatmap) to the frontend for immediate feedback.

## Learning & decision-making
- **Bandit**: Diagonal Linear Thompson Sampling (`app/ml/lints_agent.py`) with hierarchical reward on accuracy, smoothness (IKI CV + spike rate), and effective WPM.
- **Embeddings**: 16-dim PCA snippet embeddings queried through FAISS (`app/ml/vector_store.py`), stored in Postgres and exported to on-disk index/metadata. 
- **User state**: 57-dim EMA + 57-dim stddev + previous snippet embedding → 130-dim context passed to the bandit.
- **Reward**: Baseline from EMA, deltas on accuracy/smoothness/eff. WPM scaled and clipped to avoid exploding updates.
- **Persistence**: Bandit weights saved to `app/ml/lints_model.pkl`; user feature EMA/variance persisted in the `User` row.

## Metrics we expose
- `wpm`, `rawWpm`, `accuracy`, `errors` per session.
- `smoothness` from global IKI CV and spike rate; `avgIki` and spike counts underneath.
- Rollover rates overall plus `rolloverL2L`, `rolloverR2R`, `rolloverCross`; fluency scores per hand (`leftFluency`, `rightFluency`, `crossFluency`).
- `kspc`, `avgChunkLength`, speed timeline (`speedSeries`), and replay events (`replayEvents`) with chunk boundaries and rollovers.
- Best WPMs tracked per timed mode (15/30/60/120s).

## How adaptation works
- Build user context: load EMA/stddev from Postgres (or zeros), include previous snippet embedding if available.
- Thompson sample bandit weights → query vector (16-dim) → FAISS search top-k.
- Filter out current + recent snippet ids; fall back to top candidate if all filtered.
- Return chosen snippet plus predicted WPM/accuracy/consistency from the EMA vector.
- After session: compute keystroke metrics, update EMA/variance, compute reward vs pre-session EMA, and update the bandit.

## Data & pipeline
- Postgres stores users, snippets (with embeddings), sessions, and keystroke events.
- Keystroke ingestion (`/api/sessions`) computes IKIs, spike rate, rollovers, transitions, and per-char stats via `UserFeatureExtractor`.
- FAISS index build: `python -m scripts.build_faiss_index` from `backend/` (uses snippet embeddings already in DB).
- Snippet/telemetry utilities live in `backend/scripts/` (seed data, init db, condense embeddings, etc.).

## Evaluation & correctness
- Reward grounded in deltas vs EMA baselines to avoid runaway difficulty; clip deltas to keep updates bounded.
- Smoothness and fluency come from raw IKIs and rollover detection (press before prior keyup) per session.
- Smoke tests: `cd backend && pytest` for retrieval and difficulty routines.
- Health check: `GET /api/health` and FastAPI docs at `/docs`.

## Observability (plan)
- **What we log today**: session ingestion stats (WPM, accuracy), rollout metrics (IKI CV, spike rate, rollover), and agent rewards on update paths.
- **What we would alert on**: high 5xx on `/sessions` or `/snippets/retrieve`, empty FAISS responses, reward collapse (nan/zero drift), spike-rate blowups, and long-tail latencies on snippet retrieval.
- **Product analytics**: Umami snippet in `frontend/index.html` (optional, self-hosted) for basic page views; expand to event-level later.

## API/type contracts (drift plan)
- Backend OpenAPI is the source of truth. Frontend TS types in `src/types` are maintained manually and reviewed alongside backend changes.
- To avoid drift: keep PRs that touch API schemas paired with TS type updates; rely on backend validation errors during dev builds.
- At scale: generate a typed client from OpenAPI (e.g., openapi-typescript) and gate merges on type checks.

## Running locally
- Prereqs: Python 3.11+, Node 20+, Postgres 15+.
- Backend:
  ```bash
  cd backend
  cp .env.example .env   # edit SECRET_KEY and DATABASE_URL if needed
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  alembic upgrade head
  python -m scripts.init_db            
  python -m scripts.build_faiss_index  
  uvicorn app.main:app --host 0.0.0.0 --port 8000
  ```
- Frontend:
  ```bash
  cd frontend
  npm install
  VITE_API_URL=http://localhost:8000/api npm run dev -- --host
  ```
- Environment knobs: `DATABASE_URL`, `SECRET_KEY`, `FAISS_INDEX_PATH`/`SNIPPET_METADATA_PATH` (see `app/config.py`).

## Running via Docker
- Compose for dev: `docker-compose up --build` (services: Postgres, backend on :8000, frontend on :5173).
- Trainer profile for offline scripts: `docker-compose --profile train up trainer`.
- Data persistence: FAISS index/metadata mounted at `backend/data`.

## Deployment and URLs
- Backend base: `http://localhost:8000/api`; health at `/health`; docs at `/docs`.
- Frontend dev: `http://localhost:5173`; set `VITE_API_URL` for other environments.
- If serving statics elsewhere, proxy `/api` to the FastAPI service and keep `/docs` protected.

## Design principles
- Measurement before magic: every adaptation step is tied to observable keystroke metrics.
- Bounded exploration: Thompson sampling with minimum variance and clipped rewards.
- Fast feedback: per-session analytics (speed series, replay events) surface what the model sees.
- Resilient defaults: cold start uses neutral EMA baselines and filters recent snippets.

## FAQ
- **What happens if FAISS is empty?** We retry with a random vector; if still empty, return 404.
- **Where is state stored?** User EMA/variance in Postgres; bandit weights in `app/ml/lints_model.pkl`; FAISS index/metadata in `backend/data`.
- **How do I change snippet data?** Update rows in Postgres, then rerun `python -m scripts.build_faiss_index`.
- **How do I point the frontend elsewhere?** Set `VITE_API_URL` before `npm run dev/build`.

## Model & data lifecycle
- See `docs/model_lifecycle.md` for FAISS rebuilds, bandit state persistence, cold start defaults, and rollback guidance.

## Project structure
```
.
├── alembic.ini
├── backend
│   ├── alembic
│   │   ├── env.py
│   │   ├── README
│   │   ├── script.py.mako
│   │   └── versions
│   ├── alembic.ini
│   ├── app
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── core
│   │   │   └── security.py
│   │   ├── database.py
│   │   ├── generator
│   │   │   ├── __init__.py
│   │   │   ├── build_ngrams.py
│   │   │   ├── config.py
│   │   │   ├── data
│   │   │   ├── enhance_wordlist.py
│   │   │   ├── generate.py
│   │   │   └── populate_db.py
│   │   ├── main.py
│   │   ├── ml
│   │   │   ├── __init__.py
│   │   │   ├── feature_aggregator.py
│   │   │   ├── lints_agent.py
│   │   │   ├── lints_model.pkl
│   │   │   ├── snippet_features.py
│   │   │   ├── snippet_pca_16.pkl
│   │   │   ├── user_features.py
│   │   │   └── vector_store.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── db_models.py
│   │   │   └── schema.py
│   │   ├── routers
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── health.py
│   │   │   ├── profile_merge.py
│   │   │   ├── sessions.py
│   │   │   ├── snippets.py
│   │   │   └── users.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── metrics.py
│   │       └── preprocessing.py
│   ├── data
│   │   ├── bigram_freqs.json
│   │   ├── english_10k.json
│   │   ├── english_10k_enriched.json
│   │   ├── faiss_index.bin
│   │   ├── snippet_metadata.json
│   │   ├── snippets.json
│   │   ├── trigram_freqs.json
│   │   └── word_features.json
│   ├── Dockerfile
│   ├── Dockerfile.train
│   ├── requirements.txt
│   └── scripts
│       ├── analyze_data.py
│       ├── build_faiss_index.py
│       ├── cleanup_snippets.py
│       ├── condense_snippet_embeddings.py
│       ├── debug_difficulty.py
│       ├── init_db.py
│       ├── prepare_telemetry_batches.py
│       └── seed_data.py
├── docker-compose.yml
├── docs
│   ├── infra.md
│   └── model_lifecycle.md
├── frontend
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── postcss.config.js
│   ├── public
│   ├── src
│   │   ├── api
│   │   │   └── client.ts
│   │   ├── App.tsx
│   │   ├── components
│   │   │   ├── AuthModal.tsx
│   │   │   ├── dashboard
│   │   │   │   ├── FlowRadar.tsx
│   │   │   │   ├── IkiHistogramWidget.tsx
│   │   │   │   ├── KeyboardHeatmap.tsx
│   │   │   │   ├── ReplayChunkStrip.tsx
│   │   │   │   ├── ResultsDashboard.tsx
│   │   │   │   ├── RolloverBreakdown.tsx
│   │   │   │   ├── SessionStatsWidget.tsx
│   │   │   │   ├── SkillBars.tsx
│   │   │   │   └── SpeedGraph.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── TypingZone.tsx
│   │   │   ├── TypingZoneStatsDisplay.tsx
│   │   │   └── UserMenu.tsx
│   │   ├── context
│   │   │   ├── AuthContext.tsx
│   │   │   └── SessionModeContext.tsx
│   │   ├── hooks
│   │   │   ├── useKeystrokeTracking.ts
│   │   │   ├── useTypingSession.ts
│   │   │   └── useWPMCalculation.ts
│   │   ├── index.css
│   │   ├── main.tsx
│   │   ├── pages
│   │   │   ├── AuthPage.tsx
│   │   │   ├── ChangePasswordPage.tsx
│   │   │   ├── ChangeUsernamePage.tsx
│   │   │   ├── DeleteAccountPage.tsx
│   │   │   ├── LeaderboardPage.tsx
│   │   │   ├── StatsPage.tsx
│   │   │   └── WikiPage.tsx
│   │   ├── types
│   │   │   ├── index.ts
│   │   │   └── react-calendar-heatmap.d.ts
│   │   └── utils
│   │       ├── anonymousUser.ts
│   │       ├── canvas.ts
│   │       └── storage.ts
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
├── README.md
└── screenshots
  ├── results.png
  ├── stats.png
  └── type.png
```

## Contributing

Contributions welcome! Please fork, create a feature branch, and submit a PR.
