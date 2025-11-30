# FlowType MVP: Training Pipeline & Deployment

## Architecture Summary

FlowType is a typing trainer that uses machine learning to personalize snippet selection. The MVP includes:

- **Frontend**: React + Vite (TypeScript), streams per-snippet telemetry on completion.
- **Backend**: FastAPI (Python 3.11+), ingests telemetry, stores raw payloads in `telemetry_snippet_raw` (SQLite dev, Postgres prod).
- **Training Pipeline**: Batch script to convert raw telemetry → training examples; LightGBM model for snippet ranking.
- **Async Ingestion**: Hybrid approach: sync write to DB, best-effort enqueue to RQ for background enrichment.

## Quick Start (Local Development)

### Prerequisites
- Docker and Docker Compose (for Redis, worker, full stack)
- Python 3.11+ (backend development)
- Node.js 18+ (frontend development)

### Option 1: Docker Compose (Recommended for MVP)

This runs backend, frontend, Redis, and an RQ worker in containers:

```bash
cd /home/aditya/flowtype
docker-compose up --build
```

Then visit:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

The worker will automatically process telemetry jobs enqueued to Redis.

### Option 2: Manual Local Development

1. **Backend**:
   ```bash
   cd backend
   source venv/bin/activate
   pip install -r requirements.txt
   export PYTHONPATH=$(pwd)
   
   # Run migrations (if not already applied)
   alembic -c alembic.ini upgrade head
   
   # Start backend
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Redis & Worker** (optional, for background processing):
   ```bash
   # Start Redis
   docker run -d --name flowtype-redis -p 6379:6379 redis:7
   
   # In backend directory (with venv activated):
   rq worker telemetry
   ```

## Training Pipeline

### 1. Collect Telemetry
As users complete snippets, the frontend sends per-snippet data to `POST /api/telemetry/snippet`. This gets stored in `telemetry_snippet_raw`.

### 2. Prepare Training Batches
Extract features from raw telemetry:

```bash
cd backend
source venv/bin/activate
export PYTHONPATH=$(pwd)
python scripts/prepare_telemetry_batches.py
# Output: backend/data/training_batches.json
```

### 3. Train a Model
Train a LightGBM ranker on collected telemetry:

```bash
python app/ml/train.py --model-type ranker
# Output: backend/app/ml/models/ranker.pkl
```

### 4. Deploy Model (Future)
Update the snippet ranker endpoint to use the trained model for personalization.

## Database Schema

Key tables:
- `users`: User profiles.
- `snippets`: Typing snippets (words, difficulty metadata).
- `typing_sessions`: Session summaries (WPM, accuracy, duration).
- `snippet_usage`: Per-snippet performance snapshots (WPM, accuracy at snippet time).
- `telemetry_snippet_raw`: Raw full telemetry payloads (JSON) for training/debugging.

Migrations managed by Alembic (`alembic/versions/`).

## API Endpoints

### Telemetry Ingestion
**POST** `/api/telemetry/snippet`

Request body:
```json
{
  "snippet": {
    "snippet_id": "uuid",
    "session_id": "uuid",
    "wpm": 75.5,
    "accuracy": 0.95,
    "position": 0,
    "difficulty": 3.2,
    "started_at": "2025-11-30T12:00:00",
    "completed_at": "2025-11-30T12:05:00"
  },
  "user_state": {
    "user_id": "uuid"
  }
}
```

Response:
```json
{
  "status": "ok",
  "raw_id": "uuid",
  "enqueued": true,
  "job_id": "job-uuid"
}
```

- `raw_id`: ID of the durable record in `telemetry_snippet_raw`.
- `enqueued`: True if Redis available and job queued.
- `job_id`: RQ job ID (for tracking).

## Production Deployment Checklist

### 1. Database
- [ ] Provision managed PostgreSQL (Render, Railway, AWS RDS, etc.)
- [ ] Update `DATABASE_URL` env var
- [ ] Run Alembic migrations: `alembic -c alembic.ini upgrade head`

### 2. Hosting
- [ ] **Frontend**: Deploy to Vercel, Netlify, or use container host (Render, Railway)
  - Build: `npm run build`
  - Serve: Output in `dist/`
- [ ] **Backend**: Deploy Docker image to Render, Railway, Fly, or AWS ECS
  - Update `CORS_ORIGINS` env var to include frontend domain
  - Set `DATABASE_URL` to managed Postgres
  - Set `REDIS_URL` for RQ job processing
- [ ] **Worker**: Deploy same backend image with command `rq worker telemetry` (separate service/pod)
- [ ] **Redis**: Use managed Redis (Render, Railway, AWS ElastiCache, or Upstash)

### 3. Configuration
- [ ] Environment variables:
  - `DATABASE_URL`: Postgres connection string
  - `REDIS_URL`: Redis connection string
  - `CORS_ORIGINS`: Frontend origin (e.g., `https://myapp.com`)
  - `FAISS_INDEX_PATH`, `SNIPPET_METADATA_PATH`: Paths to data files (or S3)

### 4. Monitoring & Logging
- [ ] Set up error tracking (Sentry, Rollbar)
- [ ] Configure structured logging (JSON output to stdout for container logs)
- [ ] Add metrics/APM (DataDog, New Relic, or cloud-native options)

### 5. Data Persistence
- [ ] Backup strategy for Postgres
- [ ] Archive old telemetry (move to S3/parquet for long-term storage)
- [ ] Implement data retention policy (optional export/delete for GDPR)

## Next Steps (Post-MVP)

1. **Model Iteration**:
   - Collect more telemetry (100s–1000s of interactions).
   - Train improved ranker with better features (user history, snippet context).
   - A/B test ranker variants.

2. **Scaling**:
   - Move raw telemetry to object storage (S3 + parquet) for efficient batch processing.
   - Use Kafka/Redis Streams for high-volume telemetry.
   - Distribute model serving (use FastAPI + embedding cache for lower latency).

3. **Features**:
   - User accounts + login (OAuth2 optional).
   - Snippet tagging & feedback loop.
   - Real-time leaderboards.
   - Mobile app (React Native or dedicated native).

4. **Infrastructure**:
   - Containerized deployment (Kubernetes optional).
   - Automated CI/CD (GitHub Actions → ECR → ECS/Render).
   - Infrastructure-as-code (Terraform or Pulumi).

## File Structure

```
flowtype/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI entry point
│   │   ├── database.py           # SQLAlchemy setup
│   │   ├── config.py             # Settings
│   │   ├── models/
│   │   │   ├── db_models.py      # SQLAlchemy models (User, Snippet, etc.)
│   │   │   └── schema.py         # Pydantic schemas
│   │   ├── routers/
│   │   │   ├── sessions.py       # Session endpoints
│   │   │   ├── snippets.py       # Snippet endpoints
│   │   │   ├── telemetry.py      # Telemetry ingestion (hybrid sync/async)
│   │   │   └── ...
│   │   ├── ml/
│   │   │   ├── train.py          # Training script
│   │   │   ├── ranker.py         # Ranker model
│   │   │   ├── loss_formulation.py # Custom loss functions
│   │   │   └── models/           # Saved model artifacts
│   │   └── tasks/
│   │       └── telemetry_tasks.py # RQ background job for enrichment
│   ├── scripts/
│   │   ├── prepare_telemetry_batches.py # Feature extraction & batch prep
│   │   ├── init_db.py
│   │   └── load_corpus.py
│   ├── alembic/                  # Database migrations
│   │   ├── env.py
│   │   └── versions/
│   ├── data/
│   │   ├── training_batches.json # Generated training data
│   │   ├── snippet_metadata.json
│   │   └── google-10000-english.txt
│   ├── tests/
│   ├── Dockerfile               # Backend container
│   ├── requirements.txt
│   └── venv/
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── TypingZone.tsx
│   │   │   ├── StatsPanel.tsx
│   │   │   └── ...
│   │   ├── api/
│   │   │   └── client.ts         # API client (includes sendSnippetTelemetry)
│   │   └── hooks/
│   ├── Dockerfile               # Frontend container (multi-stage build)
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── docker-compose.yml           # Local dev stack (backend, frontend, Redis, worker)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ML_MODELS.md
│   ├── DEPLOYMENT.md
│   └── ...
└── README.md (this file)
```

## Troubleshooting

### "Cannot find module 'app'" during training
Ensure `PYTHONPATH` is set:
```bash
export PYTHONPATH=$(pwd)
python app/ml/train.py
```

### Redis connection refused during telemetry POST
Set `REDIS_URL` or start Redis locally:
```bash
docker run -d --name flowtype-redis -p 6379:6379 redis:7
```
The telemetry endpoint will still work (enqueued: false), but background jobs won't process.

### Database locked error (SQLite)
Multiple processes writing to SQLite can cause locking. For production, migrate to Postgres.
For dev, ensure only one Uvicorn process is running.

### CORS errors on frontend → backend
Update `CORS_ORIGINS` in `backend/app/config.py` or set env var `CORS_ORIGINS` to include the frontend URL.

---

## Contact & Support
For questions or contributions, refer to `docs/DEVELOPMENT.md` for local setup details.
