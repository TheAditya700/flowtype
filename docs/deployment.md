# Deployment & Environment Management

## Environments

| Env   | Frontend | Backend | Database | Use Case |
|-------|----------|---------|----------|----------|
| Dev   | :5173    | :8000   | :5432    | Local development, hot-reload |
| Stage | :5174    | :8001   | :5433    | Pre-prod testing, 24h soak |
| Prod  | :5175    | :8002   | :5434    | Production |

## Data structure
```
data/
├── offline/        # Source JSONs (snippets, n-grams, word lists)
├── dev/           # Dev FAISS index + metadata
├── stage/         # Stage FAISS index + metadata + backup/
└── prod/          # Prod FAISS index + metadata + backup/
```

These directories are mirrored in MinIO/S3 buckets:
- `flowtype-offline` (shared artifacts consumed during bootstrap)
- `flowtype-dev`, `flowtype-stage`, `flowtype-prod` (environment-specific FAISS assets)

Use `python backend/scripts/migrate_data_to_minio.py` before promotions to ensure MinIO is in sync with local files, and start the matching `docker-compose` stack so the endpoint is reachable.

## Workflow

### 1. Dev
```bash
docker-compose up
# First-time only (fresh volume):
docker-compose exec backend_dev alembic upgrade head
docker-compose exec backend_dev python scripts/bootstrap_env.py --env dev
# Test locally
```

### 2. Promote to Stage
```bash
python scripts/promote_to_stage.py  # Validates, backs up, copies
docker-compose -f docker-compose.stage.yml up
docker-compose -f docker-compose.stage.yml exec backend_stage alembic upgrade head
docker-compose -f docker-compose.stage.yml exec backend_stage python scripts/bootstrap_env.py --env stage  # first-time only
curl http://localhost:8001/health
```

`promote_to_stage.py` promotes through MinIO/S3—ensure `docker-compose -f docker-compose.stage.yml up` (or an external MinIO endpoint) is running so the script can reach `flowtype-stage` and `flowtype-offline` buckets.

**Validation on promote:**
- FAISS loads, vector count = metadata count
- Metadata structure valid (id, words, difficulty)
- Auto-backup to `stage/backup/`

### 3. Promote to Prod
```bash
python scripts/promote_to_prod.py  # Requires typing 'deploy to prod'
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml exec backend_prod alembic upgrade head
docker-compose -f docker-compose.prod.yml exec backend_prod python scripts/bootstrap_env.py --env prod  # first-time only
docker logs -f flowtype_backend_prod
```

The prod promotion script mirrors stage: artifacts are copied between MinIO buckets (`flowtype-stage` → `flowtype-prod`) with automatic backups. Confirm the prod MinIO service is reachable before running it.

**Promote when:**
- Stage tested 24+ hours
- No errors in stage logs
- During maintenance window

## Rollback
```bash
cd backend/data/{stage|prod}
cp backup/faiss_index.bin.bak faiss_index.bin
cp backup/snippet_metadata.json.bak snippet_metadata.json
docker-compose -f docker-compose.{env}.yml restart backend_{env}
```

## Database migrations
```bash
# Dev
docker-compose exec backend_dev alembic upgrade head

# Stage
docker-compose -f docker-compose.stage.yml exec backend_stage alembic upgrade head

# Prod (maintenance window, after DB backup)
docker-compose -f docker-compose.prod.yml exec backend_prod alembic upgrade head
```

## Monitoring
```bash
# Health
curl http://localhost:{8000|8001|8002}/health

# Logs
docker logs -f flowtype_backend_{dev|stage|prod}

# Database
docker-compose exec postgres_{env} psql -U flowtype_{env}
```

## Quick reference
```bash
# Build index
python scripts/build_index.py --env {dev|stage|prod}

# Promote
python scripts/promote_to_stage.py
python scripts/promote_to_prod.py

# Start/stop
docker-compose [-f docker-compose.{env}.yml] up [-d]
docker-compose [-f docker-compose.{env}.yml] down
```

## Observability
- The frontend exposes `/observability` to review certainty bands, session volume, and reward trends after each deployment.
- Ports: dev :5173, stage :5174, prod :5175. Verify the matching backend (`:8000|:8001|:8002`) is healthy before checking the dashboard.
- Expect fresh data immediately after `bootstrap_env.py` or promotions once sessions flow; stale charts usually mean the MinIO artifacts or DB seed did not complete.

![Observability Dashboard](../screenshots/observability.png)
**Observability dashboard** — Single view for session health, reward stability, and certainty across environments.

## Security
- Never commit `.env.{dev|stage|prod}`
- Rotate `SECRET_KEY` regularly in prod
- Use secrets manager for prod (AWS Secrets Manager, Vault)
- Database SSL + restricted network access in prod
