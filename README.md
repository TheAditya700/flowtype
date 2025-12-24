# NerdType

**Adaptive typing practice powered by keystroke-level telemetry, contextual bandits, and interpretable motor-skill modeling.**

NerdType is a research-driven typing trainer that adapts snippet difficulty in real time using explicit, interpretable signals from keystroke dynamics rather than opaque end-to-end models.

The system is designed **interpretability-first**: every adaptation step is grounded in observable timing metrics such as inter-key intervals, rollover behavior, chunking, and error dynamics, all of which are surfaced both to the user and the learning algorithm.

---

## Why NerdType

Traditional typing trainers (Monkeytype, TypeRacer) are largely static:
- Same word lists for all users
- Little adaptation to individual weaknesses
- Optimization focused almost entirely on WPM

**NerdType adapts to how you actually type:**
- Personalized snippet selection based on a learned user skill state
- Optimization prioritizes **accuracy → smoothness → speed**, mirroring motor learning
- Rich metrics beyond WPM: IKI variance, spike rate, rollover, chunk length, per-hand fluency
- A contextual bandit continuously learns which text patterns challenge you productively

---

## Research grounding

NerdType’s metrics and adaptation loop are informed by established work in typing dynamics and motor control:

- **Yin et al., CHI 2018**  
  Large-scale analysis of 136M keystrokes showing expert performance emerges from rhythmic timing, rollover behavior, chunked motor plans, and reduced variance.

- **Logan & Crump, 2011**  
  Demonstrates hierarchical motor control in skilled typing, motivating chunk-level and fluency-based metrics.

- **Killourhy & Maxion, 2009**  
  Shows inter-key interval distributions are stable and information-rich signals for modeling skill.

These findings motivate NerdType’s focus on **timing-based, interpretable signals** rather than black-box sequence models.

---

## System overview

### Architecture
- **Backend**: FastAPI, Postgres, FAISS
- **Frontend**: React + Vite
- **Storage**: Postgres (state), MinIO (artifacts)
- **Learning**: Diagonal Linear Thompson Sampling (LinTS)

### Core loop
1. Build a **130-dim user state** (EMA skill + variance + recent difficulty context)
2. Thompson-sample bandit weights to generate a query vector
3. Retrieve candidate snippets via FAISS (16-dim PCA embedding)
4. Sample snippets probabilistically for controlled exploration
5. Collect keystroke telemetry and compute session metrics
6. Compute reward vs EMA baseline and update the bandit

---

## Representation choices

### Snippet embeddings
- Snippet features are projected to **16 dimensions via PCA**
- 16D captures ~97% variance while keeping the bandit stable and sample-efficient
- Higher dimensions increased posterior variance and destabilized exploration

### User state (130D)
- **57D EMA**: long-term skill baseline
- **57D stddev**: short-term variability / consistency
- **16D previous snippet embedding**: smooth curriculum transitions

This separation prevents overreaction to single sessions and reduces difficulty oscillation.

---

## Learning objective (hierarchical reward)

The bandit optimizes a **hierarchical reward** aligned with motor learning:

``&
R = scale * [ w1 * dA
            + w2 * (dA * dC)
            + w3 * (dA * dC * dS) ]
``&

Where:
- `dA`: accuracy delta vs EMA
- `dC`: smoothness delta (IKI CV + spike rate)
- `dS`: effective WPM delta

Low accuracy suppresses downstream rewards, preventing speed optimization at the cost of correctness.

---

## Screenshots

### Typing surface
![Typing surface](screenshots/type.png)

Adaptive snippet selection with real-time keystroke capture. Every keydown and keyup feeds IKI, rollover, and chunking metrics used by the model.

---

### Session results
![Session results](screenshots/results.png)

Post-session breakdown of WPM vs raw WPM, accuracy, smoothness, rollover, and fluency metrics derived directly from keystroke timing.

---

### Stats dashboard
![Stats dashboard](screenshots/stats.png)

Longitudinal view of speed, accuracy, and fluency trends with EMA smoothing, showing how the model perceives skill over time.

---

### Leaderboard
![Leaderboard](screenshots/leaderboard.png)

Mode-specific rankings with anonymized users and best WPMs tracked per timed mode.

---

### Reference / Wiki
![Wiki](screenshots/wiki.png)

Metric definitions, modes, and guidance to help users interpret what the system measures and why.

---

### Observability dashboard
![Observability dashboard](screenshots/observability.png)

Learning health view exposing reward stability, posterior confidence, and feature-level certainty so model behavior can be debugged in-product.

---

## Metrics exposed

Per session:
- WPM, raw WPM, accuracy, errors
- Smoothness (IKI CV + spike rate)
- Rollover rates (overall, L2L, R2R, cross)
- Per-hand fluency
- Chunk length, KSPC
- Speed timeline, replay events, keyboard heatmaps

Long-term:
- EMA skill trends
- Best WPMs per timed mode
- Bandit reward and confidence dynamics

---

## Observability

NerdType includes an **in-app observability dashboard** focused on learning health:
- Posterior confidence and convergence
- Reward stability and drift
- Session volume and latency
- Interpretable user-skill components ranked by impact, certainty, and uncertainty

This makes model behavior inspectable without digging into raw weights.

---

## Running locally

### Backend

Prereqs: Postgres + MinIO reachable at the endpoints in `backend/.env`. The easiest way is to run the dev stack services in Docker (they expose Postgres on :5432 and MinIO on :9000/:9001):

```bash
docker compose up postgres_dev minio-dev
```

Then bootstrap the backend (from the repo root):

```bash
cd backend
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
python scripts/migrate_data_to_minio.py
python scripts/bootstrap_env.py --env dev
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`migrate_data_to_minio.py` seeds the required MinIO buckets (you may see warnings for stage/prod if those stacks are offline—safe to ignore).

### Frontend

```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000/api npm run dev -- --host
```

---

## Running with Docker

```bash
docker compose up
```

- Frontend: http://localhost:5173  
- Backend: http://localhost:8000  
- MinIO: http://localhost:9001  

First-time setup (inside backend container):
```bash
docker compose exec backend_dev python scripts/migrate_data_to_minio.py
docker compose exec backend_dev alembic upgrade head
docker compose exec backend_dev python scripts/bootstrap_env.py --env dev
```

---

## Design principles

- **Measurement before magic**: every adaptation is tied to observable metrics
- **Bounded exploration**: Thompson sampling with clipped rewards
- **Interpretability first**: no opaque end-to-end policies
- **Stable learning**: EMA baselines and variance-aware updates

---

## Highlights

- Applies contextual bandits to **motor-skill learning**, not clicks or ads
- Uses keystroke-level telemetry as a first-class signal
- Demonstrates careful reward shaping and posterior diagnostics
- Treats observability as part of the product, not an afterthought

---

## Environment separation and MLOps lifecycle
- See `docs/model_lifecycle.md` for FAISS rebuilds, bandit state persistence, cold start defaults, and rollback guidance.

NerdType follows a **production-grade MLOps workflow** with explicit **dev / stage / prod isolation**. Each environment runs in its own Docker Compose stack with **separate Postgres databases, FAISS indices, and MinIO object storage**, ensuring clean experimentation and reproducibility. Model artifacts (bandit weights, FAISS indices, snippet metadata) are **versioned and promoted explicitly** via scripted pipelines rather than rebuilt implicitly. Promotion flows (`dev → stage → prod`) include validation, snapshotting, and rollback guarantees as documented in `docs/deployment.md`. This design mirrors real-world ML systems by enforcing environment isolation, controlled artifact promotion, and debuggable model evolution rather than ad-hoc retraining.

---

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
│   │       ├── 3f036435bcee_add_processed_embedding_to_snippet.py
│   │       ├── 5cbee9bf7b00_merge_heads.py
│   │       ├── 7a2ea6d32ab5_initial_schema_rebuild.py
│   │       ├── 15c3f8ba6dc1_add_username_and_hashed_password_to_.py
│   │       ├── 44abdb04cca2_add_user_stats.py
│   │       ├── 90b041a2c4e9_add_top_certain_uncertain_to_model_.py
│   │       ├── 91d4c588b03b_add_agent_observability_models.py
│   │       ├── 522fbdb2c09a_add_best_wpms_to_user_and_remove_.py
│   │       ├── a7b8c9d0e1f2_merge_heads_model_snapshots_and_agent_obs.py
│   │       ├── b49cf5793c8f_change_timestamp_to_biginteger_in_.py
│   │       ├── cafe1234abcd_remove_legacy_interactions_from_model_snapshots.py
│   │       ├── d03b63f3cd1f_refactor_user_stats_to_json_features.py
│   │       ├── d25472802491_update_typing_session_schema.py
│   │       ├── e1a4f3c2d7ab_add_weights_uri_to_model_snapshots.py
│   │       ├── eb70214b29f3_add_anonymous_user_fields.py
│   │       └── f2d3c4b1a9ce_create_model_snapshots_table.py
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
│   │   │   ├── agent_models.py
│   │   │   ├── db_models.py
│   │   │   └── schema.py
│   │   ├── routers
│   │   │   ├── __init__.py
│   │   │   ├── admin.py
│   │   │   ├── auth.py
│   │   │   ├── health.py
│   │   │   ├── observability.py
│   │   │   ├── profile_merge.py
│   │   │   ├── sessions.py
│   │   │   ├── snippets.py
│   │   │   └── users.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── compute_snapshot.py
│   │       ├── metrics.py
│   │       ├── s3_data.py
│   │       └── s3_utils.py
│   ├── data
│   │   ├── dev
│   │   │   ├── faiss_index.bin
│   │   │   └── snippet_metadata.json
│   │   ├── offline
│   │   │   ├── bigram_freqs.json
│   │   │   ├── english_10k.json
│   │   │   ├── english_10k_enriched.json
│   │   │   ├── snippet_metadata.json
│   │   │   ├── snippets.json
│   │   │   ├── trigram_freqs.json
│   │   │   └── word_features.json
│   │   ├── prod
│   │   └── stage
│   ├── Dockerfile
│   ├── Dockerfile.train
│   ├── print_routes.py
│   ├── requirements.txt
│   ├── scripts
│   │   ├── analyze_data.py
│   │   ├── bootstrap_env.py
│   │   ├── build_faiss_index.py
│   │   ├── build_index.py
│   │   ├── cleanup_snippets.py
│   │   ├── condense_snippet_embeddings.py
│   │   ├── debug_difficulty.py
│   │   ├── init_db.py
│   │   ├── migrate_data_to_minio.py
│   │   ├── prepare_telemetry_batches.py
│   │   ├── promote_to_prod.py
│   │   ├── promote_to_stage.py
│   │   └── seed_data.py
│   └── tests
│       ├── conftest.py
│       ├── test_api.py
│       ├── test_health.py
│       ├── test_lints_agent.py
│       ├── test_metrics.py
│       ├── test_observability_api.py
│       └── test_user_features.py
├── dashboard_plan.md
├── docker-compose.dev.yml
├── docker-compose.prod.yml
├── docker-compose.stage.yml
├── docker-compose.yml
├── Dockerfile.minio
├── docs
│   ├── deployment.md
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
│   │   │   ├── observability
│   │   │   │   ├── AgentEffectivenessChart.tsx
│   │   │   │   ├── FeatureImportanceWidget.tsx
│   │   │   │   ├── LearningActivityChart.tsx
│   │   │   │   ├── LearningHealthChart.tsx
│   │   │   │   ├── ObservabilityHeader.tsx
│   │   │   │   ├── PerformanceDeltasChart.tsx
│   │   │   │   └── WeightsUpdatedGauge.tsx
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
│   │   │   ├── ObservabilityPage.tsx
│   │   │   ├── StatsPage.tsx
│   │   │   └── WikiPage.tsx
│   │   ├── types
│   │   │   ├── index.ts
│   │   │   └── react-calendar-heatmap.d.ts
│   │   └── utils
│   │       ├── anonymousUser.ts
│   │       ├── canvas.ts
│   │       ├── chartUtils.ts
│   │       ├── featureNames.ts
│   │       ├── storage.ts
│   │       └── suppressRechartsWarnings.ts
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
├── README.md
└── screenshots
    ├── leaderboard.png
    ├── observability.png
    ├── results.png
    ├── stats.png
    ├── type.png
    └── wiki.png
```

## Contributing

Contributions welcome. Please fork the repository, create a feature branch, and submit a PR.
