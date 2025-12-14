# Infra notes (deploy + data)

## Database migrations
- Tooling: Alembic. Use `alembic upgrade head` during deploy to apply latest migrations.
- Suggested deploy flow:
  1) Build/publish images.
  2) Run migrations against the target DB (`alembic upgrade head`).
  3) Restart backend once migrations are applied.
- For local/dev: `cd backend && alembic upgrade head`.

## FAISS index
- When to rebuild: after changing snippet embeddings (adding/removing/updating snippets or changing embedding pipeline).
- How to rebuild: from `backend/` run `python -m scripts.build_faiss_index`.
- Runtime location: `backend/data/faiss_index.bin` and `backend/data/snippet_metadata.json`. Mount or bake these into the image.
- Config knobs: `FAISS_INDEX_PATH`, `SNIPPET_METADATA_PATH` (see `app/config.py`). Ensure the runtime paths match the mounted files.

## Bandit state
- Stored at `backend/app/ml/lints_model.pkl`. Keep it if you want to preserve explore/exploit state across deploys; reset it to cold-start.

## Deploy checklist (minimal)
- Set env: `DATABASE_URL`, `SECRET_KEY`, `FAISS_INDEX_PATH`, `SNIPPET_METADATA_PATH`.
- Run migrations: `alembic upgrade head`.
- Ensure FAISS files are present (either baked or mounted); rebuild if snippet embeddings changed.
- Start backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- Health checks: `GET /api/health`; API docs at `/docs`.
