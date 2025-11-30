# FlowType Deployment Guide

This document provides instructions for deploying the FlowType application.

## Prerequisites

*   **Railway Account:** For backend deployment.
*   **Vercel Account:** For frontend deployment.
*   **Supabase Account:** For PostgreSQL database hosting.
*   **Docker:** Installed on your local machine (for building the backend image).
*   **Git:** For version control and deployment.

## 1. Database Setup (Supabase)

1.  **Create a new project in Supabase.**
2.  **Navigate to "Database" -> "Connection String"** and copy the connection string.
3.  **Update `backend/.env`:**
    Create a `.env` file in your `backend/` directory (if you haven't already) and set your `DATABASE_URL`:
    ```
    DATABASE_URL="postgresql://postgres:[YOUR-PASSWORD]@[AWS-REGION].pooler.supabase.com:6543/postgres"
    ```
    Replace `[YOUR-PASSWORD]` and `[AWS-REGION]` with your Supabase project details.

4.  **Initialize Database Schema:**
    From your local `backend/` directory, run the initialization script:
    ```bash
    cd backend
    python scripts/init_db.py
    ```

5.  **Load Corpus and Build FAISS Index:**
    ```bash
    python scripts/load_corpus.py
    python scripts/build_faiss_index.py
    ```
    This will populate your database with snippets and create the `faiss_index.bin` and `snippet_metadata.json` files in `backend/data/`. These files will be included in your Docker image.

## 2. Backend Deployment (Railway)

The backend is designed to be deployed as a single Docker container on Railway.

1.  **Ensure your `backend/data/faiss_index.bin` and `backend/data/snippet_metadata.json` files exist** (generated in step 1.5). These are crucial for the Docker build.
2.  **Create a new project in Railway.**
3.  **Connect your GitHub repository** where your FlowType project is hosted.
4.  **Choose "Deploy from GitHub Repo"** and select your repository.
5.  **Configure the service:**
    *   **Root Directory:** Set to `backend/`.
    *   **Build Command:** Railway should automatically detect the `Dockerfile`. If not, specify `docker build -t flowtype-backend .`.
    *   **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (Railway injects the `PORT` environment variable).
    *   **Environment Variables:** Add your `DATABASE_URL` from Supabase. You might also need to add `CORS_ORIGINS` if your frontend is hosted on a different domain (e.g., `https://your-frontend-domain.vercel.app`).
        ```
        DATABASE_URL=postgresql://...
        CORS_ORIGINS=http://localhost:5173,https://your-frontend-domain.vercel.app
        ```
6.  **Deploy the service.** Railway will build the Docker image and deploy your FastAPI application.

## 3. Frontend Deployment (Vercel)

The frontend is a React application built with Vite, suitable for static deployment on Vercel.

1.  **Create a new project in Vercel.**
2.  **Connect your GitHub repository** where your FlowType project is hosted.
3.  **Choose "Import Git Repository"** and select your repository.
4.  **Configure the project:**
    *   **Root Directory:** Set to `frontend/`.
    *   **Framework Preset:** Select `Vite`.
    *   **Build Command:** `npm run build`
    *   **Output Directory:** `dist`
    *   **Environment Variables:** Add `VITE_API_URL` pointing to your deployed Railway backend URL.
        ```
        VITE_API_URL=https://your-railway-backend.up.railway.app/api
        ```
5.  **Deploy the project.** Vercel will build and deploy your React application.

## Post-Deployment

*   **Verify Backend:** Access your Railway deployment URL (e.g., `https://your-railway-backend.up.railway.app/docs`) to check the FastAPI documentation and ensure endpoints are working.
*   **Verify Frontend:** Access your Vercel deployment URL and test the application end-to-end.
*   **CORS:** Ensure your `CORS_ORIGINS` environment variable on Railway includes your Vercel frontend domain to prevent cross-origin issues.
