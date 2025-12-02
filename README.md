# FlowType - Adaptive Typing Practice Application

FlowType is a full-stack web application designed to help users improve their typing skills. It uses a machine learning backend (Two-Tower Recommendation System) to adapt the difficulty of typing challenges to the user's skill level, aiming to create a "flow" state for optimal learning.

## Project Overview

**Current Status:** v0.2.0 — Two-Tower Recommendation Engine (Implemented). Moving towards RL loop.

The system features:
- **Two-Tower Architecture**: Separate User and Snippet encoders trained jointly to match users to ideal content.
- **Cold Start Handling**: Heuristic-based difficulty matching ("Zone of Proximal Development") for new users.
- **Hierarchical Loss**: Optimizes for Ranking -> Flow State -> Skill Growth -> Difficulty progression.
- **Vector Search**: FAISS-based efficient retrieval of candidate snippets.

## Architecture & Tech Stack

### Frontend
- **Framework**: React, TypeScript, Vite
- **Styling**: Tailwind CSS
- **State**: Zustand/Context
- **Features**: Real-time typing zone, live WPM/Accuracy, Session History

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: SQLite (Dev) / PostgreSQL (Prod) using SQLAlchemy & Alembic
- **ML Engine**:
    - **User Tower**: GRU (Short-term history) + MLP (Long-term stats)
    - **Snippet Tower**: MLP projecting 30-dim difficulty features
    - **Vector Store**: FAISS (FlatL2)
- **Task Queue**: Redis & RQ (for async telemetry processing)

## Setup and Running

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (optional, for Redis/Deployment)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Environment
cp .env.example .env

# Initialize Database & ML Assets
python scripts/init_db.py
python scripts/load_corpus.py
python app/ml/snippet_encoder.py  # Generate initial embeddings
python scripts/calc_difficulty.py # Calibrate difficulty scores
python scripts/build_faiss_index.py # Build vector index

# Run Server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 3. Training (Offline)

To train the ranker using collected session data:

```bash
cd backend
source venv/bin/activate
python scripts/train_ranker.py
```

## Project Structure

```
flowtype/
├── backend/
│   ├── app/
│   │   ├── main.py               # App entry point
│   │   ├── models/               # Pydantic & SQLAlchemy models
│   │   ├── routers/              # API endpoints (snippets, sessions)
│   │   ├── ml/
│   │   │   ├── user_encoder.py   # User Tower (GRU+MLP)
│   │   │   ├── snippet_tower.py  # Snippet Tower (MLP)
│   │   │   ├── ranker.py         # Two-Tower logic & Fallback
│   │   │   ├── loss_formulation.py # Hierarchical Loss
│   │   │   ├── vector_store.py   # FAISS wrapper
│   │   │   └── ...
│   │   └── ...
│   ├── scripts/                  # Maintenance & ML scripts
│   │   ├── train_ranker.py       # Training loop
│   │   ├── calc_difficulty.py    # PCA calibration
│   │   └── ...
│   └── data/                     # Local storage for index/metadata
│
├── frontend/
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── api/                  # API client
│   │   └── types/                # TypeScript interfaces
│   └── ...
└── docs/                         # Detailed documentation
```

## Documentation

- [API Documentation](docs/API.md)
- [Architecture Design](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## Contributing

Contributions are welcome! Please check the issues tab and follow standard PR workflows.
