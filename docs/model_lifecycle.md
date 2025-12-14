# Model & Data Lifecycle

This document describes how snippet embeddings, the FAISS index, and the LinTS bandit state are produced, refreshed, and rolled back.

## Components
- **Snippet embeddings**: Stored in Postgres (`snippets.embedding` and `snippets.processed_embedding`).
- **FAISS index + metadata**: On-disk at `backend/data/faiss_index.bin` and `backend/data/snippet_metadata.json`.
- **Bandit weights**: Thompson Sampling weights persisted at `backend/app/ml/lints_model.pkl`.

## Build & refresh flows
- **Initial seed**: Load snippets into Postgres (via seed/import), compute embeddings (offline script), then build FAISS.
- **FAISS rebuild**: Run from `backend/`: `python -m scripts.build_faiss_index`. This reads snippet embeddings from the DB and writes the index + metadata to `backend/data/`.
- **Bandit state**: Persisted automatically on update; stored locally at `app/ml/lints_model.pkl`. Back up this file if you want to preserve learned behavior across deploys.
- **Cold start**: If no bandit weights are present, a fresh LinTS model is initialized with neutral priors; user EMAs default to neutral baselines.

## When to rebuild FAISS
- After adding, editing, or deleting snippet embeddings in Postgres.
- After changing the embedding model or PCA dimensions.
- After bulk data cleanup that invalidates existing IDs.

## Deployment guidance
- Ship `backend/data/faiss_index.bin` and `backend/data/snippet_metadata.json` with the image, or mount a volume to provide them at runtime.
- Ensure `FAISS_INDEX_PATH` and `SNIPPET_METADATA_PATH` point to the mounted files (see `app/config.py`).
- Back up `app/ml/lints_model.pkl` if you want to keep explore/exploit state between releases.

## Rollback strategy
- **FAISS**: Keep the previous `faiss_index.bin` + metadata; if a rebuild is bad (empty index, wrong dim), swap back the prior files and restart the service.
- **Bandit**: Keep a copy of the previous `lints_model.pkl`; if rewards go off (reward collapse, over-exploitation), restore the last known-good weights.

## Validation checklist
- FAISS index shape matches embedding dimension (default 16).
- Sample retrieval works: `POST /api/snippets/retrieve` returns non-empty results.
- Bandit reward and smoothness metrics are finite after a session.
- No schema drift: embeddings exist for all active snippets.
