# Infrastructure

## Database migrations
```bash
# Apply migrations (Alembic)
docker-compose exec backend_{env} alembic upgrade head

# Local
cd backend && alembic upgrade head
```

## Multi-environment setup
- **Dev**: DB :5432, Backend :8000, Frontend :5173
- **Stage**: DB :5433, Backend :8001, Frontend :5174
- **Prod**: DB :5434, Backend :8002, Frontend :5175

Each env has:
- Isolated Postgres container
- Separate FAISS index in `data/{dev|stage|prod}/`
- Environment-specific `.env.{dev|stage|prod}`

## FAISS index management
```bash
# Build for specific env
python scripts/build_index.py --env {dev|stage|prod}

# Paths per env
data/dev/faiss_index.bin + snippet_metadata.json
data/stage/faiss_index.bin + snippet_metadata.json
data/prod/faiss_index.bin + snippet_metadata.json
```

## Config knobs
- `ENV`: dev | stage | prod
- `DATABASE_URL`: postgres connection string
- `SECRET_KEY`: JWT signing key (rotate in prod)
- `FAISS_INDEX_PATH`: path to FAISS index
- `SNIPPET_METADATA_PATH`: path to snippet metadata

## Docker
```bash
# Start environments
docker-compose up                                   # dev
docker-compose -f docker-compose.stage.yml up      # stage
docker-compose -f docker-compose.prod.yml up -d    # prod (detached)

# First-time bootstrap (fresh volumes)
docker-compose exec backend_dev alembic upgrade head && docker-compose exec backend_dev python scripts/bootstrap_env.py --env dev
docker-compose -f docker-compose.stage.yml exec backend_stage alembic upgrade head && docker-compose -f docker-compose.stage.yml exec backend_stage python scripts/bootstrap_env.py --env stage
docker-compose -f docker-compose.prod.yml exec backend_prod alembic upgrade head && docker-compose -f docker-compose.prod.yml exec backend_prod python scripts/bootstrap_env.py --env prod

# Logs
docker logs -f flowtype_backend_{env}

# Database access
docker-compose exec postgres_{env} psql -U flowtype_{env}
```

## Health checks
```bash
curl http://localhost:8000/health  # dev
curl http://localhost:8001/health  # stage
curl http://localhost:8002/health  # prod
```

## Secrets (prod)
- Rotate `SECRET_KEY` regularly
- Use secrets manager (AWS Secrets Manager, Vault)
- Never commit `.env.prod`
- Database SSL in production
