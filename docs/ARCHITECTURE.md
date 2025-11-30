# FlowType Architecture

This document outlines the high-level architecture of the FlowType application.

## Overview

FlowType is an adaptive typing application designed to help users improve their typing skills. It uses a two-tower machine learning system to recommend typing snippets that are optimally challenging for the user's current skill level, aiming to keep them in a "flow" state.

## Components

### 1. Frontend (React + TypeScript + Vite)

*   **User Interface:** Built with React, providing a responsive and interactive typing experience.
*   **State Management:** React hooks manage the application's local state, including typing progress, real-time statistics, and user input.
*   **API Client:** Handles communication with the FastAPI backend to fetch snippets and save session data.
*   **Styling:** Tailwind CSS for a utility-first approach to styling.
*   **Build Tool:** Vite for fast development and optimized builds.

### 2. Backend (FastAPI + Python)

*   **API Endpoints:** Provides RESTful APIs for snippet retrieval, session management, and user statistics.
*   **Database Interaction:** Uses SQLAlchemy ORM to interact with the PostgreSQL database.
*   **Machine Learning Core:**
    *   **User Encoder:** Transforms user performance metrics (WPM, accuracy, errors) into a vector embedding.
    *   **Snippet Encoder:** Uses `sentence-transformers` (all-MiniLM-L6-v2) to convert typing snippets into vector embeddings.
    *   **FAISS Vector Store:** An in-memory FAISS index stores snippet embeddings, enabling fast similarity search.
    *   **Difficulty Scoring:** Calculates a difficulty score for each snippet based on features like word length, rare letters, and punctuation.
    *   **Ranker:** Ranks candidate snippets retrieved from FAISS, considering both similarity to user state and adaptive difficulty logic, to select the next optimal snippet.

### 3. Database (PostgreSQL via Supabase)

*   **Users Table:** Stores basic user information (optional for MVP).
*   **Snippets Table:** Stores pre-processed typing snippets along with their calculated difficulty features.
*   **Typing Sessions Table:** Records detailed telemetry for each typing session, including WPM, accuracy, errors, keystroke events, and difficulty progression.
*   **Snippet Usage Table:** Tracks which snippets were presented during a session.

## Data Flow

1.  **User Interaction:** User types in the frontend.
2.  **Frontend State Update:** Real-time WPM, accuracy, and keystroke events are tracked.
3.  **Session Completion:** When a session ends, the frontend sends session data to the backend.
4.  **Backend Session Storage:** The backend saves the session data to PostgreSQL.
5.  **Next Snippet Request:** The frontend requests the next snippet, sending the current user state (derived from recent performance) to the backend.
6.  **User State Encoding:** The backend's user encoder transforms the user state into an embedding.
7.  **Vector Search:** The user embedding is used to query the FAISS vector store, retrieving a set of candidate snippets.
8.  **Snippet Ranking:** The ranker re-ranks these candidates, applying adaptive difficulty logic to select the most suitable snippet.
9.  **Snippet Delivery:** The selected snippet is sent back to the frontend for the user to type.

## Machine Learning Pipeline

1.  **Offline Preprocessing:**
    *   Raw word lists are cleaned and segmented into snippets.
    *   Each snippet's difficulty features (e.g., average word length, rare letter count) are calculated.
    *   Snippet text is encoded into embeddings using `sentence-transformers`.
    *   A FAISS index is built from these snippet embeddings and saved to disk along with metadata.
2.  **Online Inference (Snippet Retrieval):**
    *   User's real-time performance metrics are used to create a user state vector.
    *   This user state vector queries the FAISS index to find similar snippets.
    *   A ranking function then selects the best snippet, considering both similarity and adaptive difficulty.

## Deployment Strategy

*   **Backend:** Packaged as a single Docker container and deployed to Railway.
*   **Frontend:** Built as static assets and deployed to Vercel.
*   **Database:** PostgreSQL hosted on Supabase.
