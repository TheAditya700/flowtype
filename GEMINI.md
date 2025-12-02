# Gemini Context: FlowType Project

This document provides context for the FlowType project, an adaptive typing practice application.

## Project Overview

FlowType is a full-stack web application designed to help users improve their typing skills. It uses a machine learning backend to adapt the difficulty of typing challenges to the user's skill level, aiming to create a "flow" state for optimal learning.

**Current Status:** v0.2.0 — Engine Rewrite Milestone (Completed). Moving towards a Two-Tower architecture.

The project is a monorepo with two main parts:
1.  **`backend`**: A Python-based API built with **FastAPI**. It handles user sessions, statistics, and the core machine learning logic for snippet selection.
2.  **`frontend`**: A **React** application built with **Vite** and written in **TypeScript**. It provides the user interface for typing, displaying stats, and viewing session history.

### Architecture & Tech Stack

-   **Frontend**: React, TypeScript, Vite, Tailwind CSS, Zustand.
-   **Backend**: FastAPI, Python 3.11, SQLAlchemy, Alembic, Redis (for RQ task queue).
-   **Database**: SQLite (local) / PostgreSQL (production).
-   **ML/Vector Search**:
    -   **Engineered Features**: Custom difficulty feature extractor (50+ ergonomic & linguistic features) and normalized feature vectors.
    -   **Vector Search**: `faiss-cpu` for in-memory vector search.
    -   **Embeddings**: `sentence-transformers` for semantic embeddings, plus explicit 30-dim difficulty vectors.
    -   **Architecture**: Two-tower model (User Tower + Snippet Tower). Currently implementing the User Tower (GRU-based).
-   **Deployment**: Dockerized backend.

## Key Features (v0.2.0)

-   **Snippet Generation**: Pipeline with weighted n-gram generation (Zipf frequencies) and synthetic variety.
-   **Difficulty Engine**: Calculates 50+ features per snippet, vectorizes, and normalizes them.
-   **Telemetry**: Ingestion of raw keystroke data for future ML training (fatigue/rhythm modeling).
-   **Session Tracking**: WPM, accuracy, backspace rate, rolling difficulty estimation.

## Building and Running

### Backend (FastAPI)

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up environment variables:**
    Copy `.env.example` to `.env` and configure `DATABASE_URL` and Redis connection.
    ```bash
    cp .env.example .env
    ```
5.  **Prepare data and ML models:**
    Run the initialization scripts.
    ```bash
    python scripts/init_db.py
    python scripts/load_corpus.py
    python scripts/build_faiss_index.py
    ```
6.  **Run the development server:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

### Frontend (React)

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```
2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```
3.  **Run the development server:**
    ```bash
    npm run dev
    ```

## Development Conventions

-   **Monorepo Structure**: `backend` and `frontend` directories.
-   **Backend Organization**:
    -   `app/ml`: Core ML logic (encoders, rankers, feature extraction).
    -   `app/generator`: Snippet generation pipeline.
    -   `app/routers`: API endpoints.
    -   `app/models`: Pydantic schemas and ORM models.
-   **Type Safety**: Strict Python type hints and TypeScript.
-   **Database**: Alembic for migrations.
-   **Task Queue**: RQ and Redis for background tasks (e.g., telemetry processing).