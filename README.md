# FlowType - Adaptive Typing Practice Application

This project aims to build an adaptive typing practice application with a focus on learning the full ML systems stack. It features a React frontend, a FastAPI backend, PostgreSQL for data storage, and FAISS for vector search.

## Project Structure

```
flowtype/
├── README.md
├── .gitignore
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                      # FastAPI app entry
│   │   ├── config.py                    # Settings (DB URLs, etc.)
│   │   ├── database.py                  # SQLAlchemy setup
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── schema.py                # Pydantic models (API contracts)
│   │   │   └── db_models.py             # SQLAlchemy ORM models
│   │   │
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   ├── health.py                # Health check endpoint
│   │   │   ├── snippets.py              # GET /snippets/retrieve
│   │   │   ├── sessions.py              # POST /sessions
│   │   │   └── users.py                 # GET /users/{id}/stats
│   │   │
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── user_encoder.py          # User state → embedding
│   │   │   ├── snippet_encoder.py       # Snippet text → embedding
│   │   │   ├── difficulty.py            # Difficulty scoring logic
│   │   │   ├── ranker.py                # Flow-based ranking
│   │   │   └── vector_store.py          # FAISS wrapper class
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── preprocessing.py         # Text cleaning
│   │       └── metrics.py               # WPM, accuracy calculations
│   │
│   ├── scripts/
│   │   ├── init_db.py                   # Create tables
│   │   ├── load_corpus.py               # Load words into DB
│   │   ├── build_faiss_index.py         # Precompute embeddings
│   │   └── seed_data.py                 # Sample data for testing
│   │
│   ├── data/
│   │   └── google-10000-english.txt     # Raw word list
│   │
│   ├── tests/
│   │   ├── test_retrieval.py
│   │   └── test_difficulty.py
│   │
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                      # Main app component
│   │   ├── main.tsx                     # Entry point
│   │   │
│   │   ├── components/
│   │   │   ├── TypingZone.tsx           # Core typing interface
│   │   │   ├── WordDisplay.tsx          # Show current/next words
│   │   │   ├── StatsPanel.tsx           # Live WPM/accuracy
│   │   │   ├── ResultCard.tsx           # Post-session summary
│   │   │   ├── ProgressChart.tsx        # Difficulty over time
│   │   │   └── SessionHistory.tsx       # Past sessions list
│   │   │
│   │   ├── hooks/
│   │   │   ├── useTypingSession.ts      # Main typing logic
│   │   │   ├── useKeystrokeTracking.ts  # Capture keystrokes
│   │   │   └── useWPMCalculation.ts     # Real-time WPM
│   │   │
│   │   ├── api/
│   │   │   └── client.ts                # API calls to backend
│   │   │
│   │   ├── types/
│   │   │   └── index.ts                 # TypeScript interfaces
│   │   │
│   │   └── utils/
│   │       ├── storage.ts               # localStorage helpers
│   │       └── canvas.ts                # Result card rendering
│   │
│   ├── public/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
└── docs/
    ├── API.md                           # API documentation
    ├── ARCHITECTURE.md                  # System design doc
    └── DEPLOYMENT.md                    # Deploy instructions
```

## Setup and Running

### Backend

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```
2.  **Create a Python virtual environment and activate it:**
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
5.  **Initialize the database:**
    ```bash
    python scripts/init_db.py
    ```
6.  **Load the word corpus and build the FAISS index:**
    ```bash
    python scripts/load_corpus.py
    python scripts/build_faiss_index.py
    ```
7.  **Run the FastAPI application:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The API will be available at `http://localhost:8000`.

### Frontend

1.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```
2.  **Install Node.js dependencies:**
    ```bash
    npm install
    # or yarn install
    ```
3.  **Run the development server:**
    ```bash
    npm run dev
    # or yarn dev
    ```
    The frontend will be available at `http://localhost:5173` (or another port if 5173 is in use).

## Key Features

*   **Adaptive Difficulty:** Adjusts typing challenges based on user performance.
*   **Two-Tower Retrieval:** Uses machine learning to select optimal snippets.
*   **Real-time Stats:** Displays Words Per Minute (WPM) and accuracy live.
*   **Session Tracking:** Records detailed session data for analysis and progress tracking.
*   **Shareable Results:** Generate image cards of your typing performance.

## Deployment

*   **Backend:** Designed for deployment on Railway using the provided `Dockerfile`.
*   **Frontend:** Designed for static deployment on Vercel.

## Documentation

Complete documentation is available in the `/docs/` folder:

- **[API.md](docs/API.md)** - REST API endpoint documentation
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and data flow
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment instructions for Railway and Vercel
- **[DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development setup and best practices
- **[ML_MODELS.md](docs/ML_MODELS.md)** - Detailed ML architecture and models
- **[TESTING.md](docs/TESTING.md)** - Testing strategies and guidelines
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Project roadmap and status

## Contributing

Feel free to contribute to the development of FlowType! 

Please see [DEVELOPMENT.md](docs/DEVELOPMENT.md) for:
- Development environment setup
- Code standards and conventions
- Testing requirements
- Pull request guidelines
