# Gemini Context: FlowType Project

This document provides context for the FlowType project, an adaptive typing practice application.

## Project Overview

FlowType is a full-stack web application designed to help users improve their typing skills. It uses a **Hierarchical Typing Outcome Model (HTOM)** to predict user performance and adapt the difficulty of typing challenges. The goal is to find the "flow" state where the user is challenged but not overwhelmed, fostering accuracy, consistency, and then speed.

**Current Status:** v0.3.0 — HTOM Architecture Implementation.

The project is a monorepo with two main parts:
1.  **`backend`**: A Python-based API built with **FastAPI**. It handles user sessions, statistics, and the core ML logic for snippet selection.
2.  **`frontend`**: A **React** application built with **Vite** and written in **TypeScript**. It provides the user interface for typing, displaying stats, and viewing session history.

### Architecture & Tech Stack

-   **Frontend**: React, TypeScript, Vite, Tailwind CSS, Recharts.
-   **Backend**: FastAPI, Python 3.11, SQLAlchemy, Alembic, Redis.
-   **Database**: SQLite (local) / PostgreSQL (production).
-   **ML Architecture: Hierarchical Typing Outcome Model (HTOM)**
    -   **Concept**: Motor learning follows a hierarchy: Accuracy → Consistency → Speed.
    -   **Goal**: Predict if a (User, Snippet) pair will result in:
        1.  Success (Accuracy ≥ threshold)
        2.  Consistency (Smooth IKIs | Success)
        3.  Speed Gain (WPM Increase | Success, Consistency)
    -   **User Tower**:
        -   **Inputs**: Global skill indicators (smoothness, rollover, hand fluency, etc.) + Pooled recent snippet history (last ~10 runs).
        -   **Model**: Small MLP (Linear → ReLU → Linear → LayerNorm) producing a 32D vector.
    -   **Snippet Tower**:
        -   **Inputs**: Manual features (hand-heaviness, rollover counts, IKI difficulty, chunk complexity, etc.).
        -   **Model**: Small MLP projecting features to a 32D vector.
    -   **Prediction Heads**:
        -   Concatenates User and Snippet vectors.
        -   **Accuracy Head**: `p_acc` (Sigmoid)
        -   **Consistency Head**: `p_cons` (Sigmoid, conditional on acc)
        -   **Speed Head**: `p_wpm` (Sigmoid, conditional on acc + cons)
    -   **Loss**: Supervised Binary Cross Entropy (BCE) on historical logs.
    -   **Inference**: Selects snippets maximizing the weighted expected improvement.

## Key Features

-   **Snippet Generation**: Pipeline with weighted n-gram generation and synthetic variety.
-   **Feature Extraction**:
    -   **Snippet**: 15-20 dimensional vector (ergonomic & linguistic).
    -   **User**: 60+ dimensional vector (global stats + recent history pooling).
-   **Dashboard**: Interactive visualization of "Flow", Speed, Heatmaps, and Replay analysis.

## Development Conventions

-   **Monorepo Structure**: `backend` and `frontend` directories.
-   **Backend Organization**:
    -   `app/ml`: Core ML logic (encoders, HTOM model, feature extraction).
    -   `app/routers`: API endpoints.
    -   `app/models`: Pydantic schemas and ORM models.
-   **Type Safety**: Strict Python type hints and TypeScript.
-   **Database**: Alembic for migrations.
