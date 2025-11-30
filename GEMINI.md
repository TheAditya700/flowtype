# Gemini Context: FlowType Project

This document provides context for the FlowType project, an adaptive typing practice application.

## Project Overview

FlowType is a full-stack web application designed to help users improve their typing skills. It uses a machine learning backend to adapt the difficulty of typing challenges to the user's skill level, aiming to create a "flow" state for optimal learning.

The project is a monorepo with two main parts:
1.  **`backend`**: A Python-based API built with **FastAPI**. It handles user sessions, statistics, and the core machine learning logic for snippet selection.
2.  **`frontend`**: A **React** application built with **Vite** and written in **TypeScript**. It provides the user interface for typing, displaying stats, and viewing session history.

### Architecture & Tech Stack

-   **Frontend**: React, TypeScript, Vite, Tailwind CSS
-   **Backend**: FastAPI, Python 3.11
-   **Database**: PostgreSQL (intended for use with Supabase)
-   **ML/Vector Search**:
    -   `sentence-transformers` (`all-MiniLM-L6-v2` model) for creating text embeddings from typing snippets.
    -   `faiss-cpu` for in-memory vector search to find relevant snippets.
    -   The architecture is a **two-tower model**, encoding both user state and snippets into vector embeddings to find the best match.
-   **Deployment**: The backend is containerized with a `Dockerfile` for deployment on services like Railway. The frontend is a static build intended for platforms like Vercel.

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
    Copy `.env.example` to `.env` and fill in your PostgreSQL `DATABASE_URL`.
    ```bash
    cp .env.example .env
    ```
5.  **Prepare data and ML models:**
    Run the following scripts to initialize the database, load the word corpus, and build the FAISS vector index.
    ```bash
    python scripts/init_db.py
    python scripts/load_corpus.py
    python scripts/build_faiss_index.py
    ```
6.  **Run the development server:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The API will be available at `http://localhost:8000`.

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
    The frontend will be available at `http://localhost:5173`.

## Development Conventions

-   **Monorepo Structure**: All code is contained in the `backend` and `frontend` directories.
-   **Backend Structure**: The FastAPI application follows a standard structure, separating concerns into `routers`, `models`, `ml`, and `utils`. Database ORM models (`db_models.py`) are kept separate from Pydantic API schemas (`schema.py`).
-   **Frontend Structure**: The React application uses a component-based architecture with `components`, `hooks`, and `api` directories.
-   **Data Scripts**: The `backend/scripts` directory contains essential one-off scripts for data preparation and database initialization. These must be run before the application can function correctly.
-   **Typing**: The project uses Python type hints in the backend and is written entirely in TypeScript on the frontend, indicating a strong preference for type safety.
-   **Containerization**: The backend is designed to be run inside a Docker container, as defined by the `Dockerfile`. The container pre-downloads the sentence-transformer model to avoid a cold start.
