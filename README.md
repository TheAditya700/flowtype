# FlowType - Adaptive Typing Practice Application

FlowType is a full-stack web application designed to help users improve their typing skills with adaptive difficulty matching and performance analytics.

## Project Overview

**Current Status:** v0.3.0 — MVP Release

The system features:
- **Adaptive Difficulty**: ML-powered snippet selection based on user skill level
- **Multiple Game Modes**: 15s, 30s, 60s, 120s timed modes + free mode
- **Performance Tracking**: Real-time WPM, accuracy, heatmaps, leaderboards
- **User Accounts**: Registration, authentication, account management
- **Flow Detection**: AFK detection after 5s of inactivity
- **Analytics Dashboard**: Historical stats, skill progression, keyboard heatmaps

## Architecture & Tech Stack

### Frontend
- **Framework**: React 18, TypeScript, Vite
- **Styling**: Tailwind CSS
- **State Management**: React Context
- **Charts**: Recharts (activity heatmap, progress tracking)
- **Pages**: Type, Stats, Leaderboard, Wiki, Auth, Account Management

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: SQLite (Development) / PostgreSQL (Production) with SQLAlchemy & Alembic
- **ML Engine**:
    - **Feature Extraction**: 50+ linguistic & ergonomic features per snippet
    - **User Features**: GRU-based session history, EMA rolling stats
    - **Vector Store**: FAISS for efficient snippet retrieval
    - **RL Agent**: LinTS contextual bandit for difficulty adaptation
- **Authentication**: JWT-based with password hashing

## Features

### User-Facing
- 📝 Real-time typing with live WPM/accuracy
- 🎮 Multiple timed modes (15s-120s) + free mode
- 📊 Session history and lifetime stats
- 🔥 Keyboard heatmap showing weak keys
- 🏆 Leaderboards (all-time by mode)
- 👤 Account management (change username/password, delete account)
- 🚫 AFK detection (5s inactivity timeout)
- 🌐 Wiki reference page
- 🔐 User authentication & anonymous mode

### Technical
- Snippet difficulty calibration via PCA
- Weighted n-gram integration for challenge variety
- Keystroke telemetry collection
- Session-based reward calculation
- Best WPM tracking (timed modes only, not free mode)

## Setup and Running

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm or yarn

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Initialize Database
python scripts/init_db.py

# Build FAISS index
python scripts/build_faiss_index.py

# Run Server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to access the app.

## Project Structure

```
flowtype/
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI entry point
│   │   ├── config.py             # Configuration
│   │   ├── database.py           # SQLAlchemy setup
│   │   ├── models/
│   │   │   ├── db_models.py      # SQLAlchemy models
│   │   │   └── schema.py         # Pydantic schemas
│   │   ├── routers/              # API endpoints
│   │   │   ├── auth.py           # Authentication + account management
│   │   │   ├── snippets.py       # Snippet retrieval
│   │   │   ├── sessions.py       # Session recording
│   │   │   ├── users.py          # User stats & leaderboard
│   │   │   └── health.py         # Health checks
│   │   ├── ml/
│   │   │   ├── user_features.py  # User feature extraction
│   │   │   ├── snippet_features.py # Snippet feature computation
│   │   │   ├── lints_agent.py    # RL agent for difficulty selection
│   │   │   ├── vector_store.py   # FAISS wrapper
│   │   │   └── feature_aggregator.py # Stats aggregation
│   │   ├── core/
│   │   │   └── security.py       # JWT & password utilities
│   │   ├── generator/            # Data generation scripts
│   │   └── utils/
│   ├── scripts/                  # Maintenance scripts
│   │   ├── init_db.py            # Database initialization
│   │   ├── build_faiss_index.py  # Vector index builder
│   │   ├── seed_data.py          # Populate initial data
│   │   └── ...
│   └── data/                     # Local FAISS index & metadata
│
├── frontend/
│   ├── src/
│   │   ├── pages/                # Page components (Type, Stats, etc.)
│   │   ├── components/           # Reusable UI components
│   │   ├── context/              # React Context (Auth, SessionMode)
│   │   ├── hooks/                # Custom React hooks
│   │   ├── api/                  # API client
│   │   ├── types/                # TypeScript interfaces
│   │   └── utils/                # Utilities
│   └── ...
│
├── docs/                         # Detailed documentation
├── alembic.ini                   # Database migrations
└── docker-compose.yml            # Docker setup
```

## API Endpoints

### Authentication
- `POST /api/auth/register` — Register new user
- `POST /api/auth/token` — Login
- `GET /api/auth/users/me` — Get current user
- `PUT /api/auth/users/change-username` — Change username
- `PUT /api/auth/users/change-password` — Change password
- `DELETE /api/auth/users/delete-account` — Delete account

### Snippets & Sessions
- `POST /api/snippets/retrieve` — Get next snippet (adaptive)
- `POST /api/sessions` — Save completed session
- `GET /api/users/leaderboard` — Get leaderboard

### Stats
- `GET /api/users/{userId}/profile` — Get user profile
- `GET /api/users/{userId}/stats/detail` — Get detailed stats

## Documentation

- [API Documentation](docs/API.md)
- [Architecture Design](docs/ARCHITECTURE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Project Status & Roadmap](docs/PROJECT_STATUS.md)

## Development

### Build Frontend
```bash
cd frontend
npm run build
```

### Run Tests
```bash
cd backend
pytest
```

### Database Migrations
```bash
cd backend
alembic upgrade head  # Apply migrations
alembic revision --autogenerate -m "description"  # Create migration
```

## Future Roadmap

- [ ] Advanced analytics (skill by key, finger usage patterns)
- [ ] Custom wordlists & challenge creation
- [ ] Social features (teams, challenges)
- [ ] Mobile app (React Native)
- [ ] Enhanced RL loop with telemetry feedback
- [ ] Real-time multiplayer typing

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please fork, create a feature branch, and submit a PR.
