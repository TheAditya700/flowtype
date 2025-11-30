# FlowType Development Guide

This document provides guidelines and best practices for developing FlowType.

## Development Environment Setup

### Prerequisites

- Python 3.11+
- Node.js 16+
- PostgreSQL 13+ (or Supabase account)
- Git

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your DATABASE_URL
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Database Initialization

```bash
cd backend
python scripts/init_db.py
python scripts/load_corpus.py
python scripts/build_faiss_index.py
```

## Running the Application

### Start Backend (Terminal 1)

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API documentation available at: `http://localhost:8000/docs`

### Start Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

Frontend available at: `http://localhost:5173`

## Project Structure & Conventions

### Backend Organization

```
backend/app/
├── main.py              # FastAPI app initialization
├── config.py            # Settings and environment variables
├── database.py          # SQLAlchemy setup
├── models/              # Database and API models
│   ├── db_models.py     # SQLAlchemy ORM models
│   └── schema.py        # Pydantic request/response schemas
├── routers/             # API endpoint handlers
│   ├── health.py
│   ├── snippets.py
│   ├── sessions.py
│   └── users.py
├── ml/                  # Machine learning modules
│   ├── user_encoder.py  # User state → embedding
│   ├── snippet_encoder.py
│   ├── difficulty.py
│   ├── ranker.py
│   └── vector_store.py
└── utils/               # Utility functions
    ├── preprocessing.py
    └── metrics.py
```

**Key Conventions:**
- **Separation of Concerns:** Database models in `db_models.py`, API schemas in `schema.py`
- **Stateless Design:** Routers should not maintain state
- **Dependency Injection:** Use FastAPI's dependency system for shared resources
- **Type Hints:** All functions must have type hints

### Frontend Organization

```
frontend/src/
├── App.tsx              # Main component
├── main.tsx             # Entry point
├── components/          # Reusable React components
├── hooks/               # Custom React hooks
├── api/                 # API client
├── types/               # TypeScript interfaces
└── utils/               # Utility functions
```

**Key Conventions:**
- **Component Naming:** PascalCase (e.g., `TypingZone.tsx`)
- **File Organization:** Co-locate styles with components when using CSS modules
- **Hooks:** Extract complex logic into custom hooks
- **Types:** Define all prop types as interfaces in `types/index.ts`

## Code Standards

### Python

- **Linting:** Use `black` for formatting (standard in this project)
- **Type Hints:** All functions must include type hints
- **Docstrings:** Use Google-style docstrings for public functions

Example:
```python
def calculate_wpm(words_typed: int, duration_seconds: float) -> float:
    """
    Calculate words per minute.
    
    Args:
        words_typed: Number of words typed
        duration_seconds: Duration in seconds
        
    Returns:
        Words per minute as float
    """
    if duration_seconds == 0:
        return 0.0
    return (words_typed / duration_seconds) * 60
```

### TypeScript

- **Strict Mode:** Enabled in `tsconfig.json`
- **Naming Conventions:** camelCase for variables, PascalCase for types/components
- **Null Safety:** Avoid `any` type, use proper typing

Example:
```typescript
interface UserMetrics {
  wpm: number;
  accuracy: number;
  errors: number;
}

function calculateAccuracy(
  correctChars: number,
  totalChars: number
): number {
  return totalChars === 0 ? 0 : correctChars / totalChars;
}
```

## Testing

### Running Backend Tests

```bash
cd backend
pytest tests/
```

### Writing Backend Tests

```python
# tests/test_difficulty.py
import pytest
from app.ml.difficulty import calculate_difficulty

def test_calculate_difficulty_short_words():
    """Test difficulty calculation for snippets with short words."""
    snippet = "the cat sat"
    difficulty = calculate_difficulty(snippet)
    assert 1.0 <= difficulty <= 2.0  # Easy snippet
```

### Frontend Testing

```bash
cd frontend
npm run test  # If test script is configured
```

## Working with the Database

### Adding a New Column

1. **Update the ORM model** in `backend/app/models/db_models.py`
2. **Create a migration** (if using Alembic):
   ```bash
   cd backend
   alembic revision --autogenerate -m "Add new column"
   alembic upgrade head
   ```
3. **Update the Pydantic schema** in `backend/app/models/schema.py`

### Adding a New Table

1. **Define the SQLAlchemy model** in `backend/app/models/db_models.py`
2. **Create migration** (if using Alembic)
3. **Add Pydantic schema** in `backend/app/models/schema.py`
4. **Initialize in script** if needed (in `backend/scripts/`)

## API Development

### Adding a New Endpoint

1. **Create or update router** in `backend/app/routers/`
2. **Define Pydantic schemas** (request/response) in `backend/app/models/schema.py`
3. **Implement handler** with proper type hints:

```python
from fastapi import APIRouter, HTTPException
from app.models.schema import SnippetRequest, SnippetResponse

router = APIRouter()

@router.post("/retrieve", response_model=list[SnippetResponse])
async def retrieve_snippets(request: SnippetRequest):
    """
    Retrieve snippets based on user state.
    
    Args:
        request: User state and preferences
        
    Returns:
        List of recommended snippets
        
    Raises:
        HTTPException: If snippet retrieval fails
    """
    # Implementation
    pass
```

4. **Register router** in `backend/app/main.py`

### API Response Format

All endpoints should follow a consistent response format:

**Success (200):**
```json
{
  "data": {...},
  "status": "success"
}
```

**Error (4xx, 5xx):**
```json
{
  "detail": "Error message",
  "status": "error"
}
```

## Machine Learning Development

### Modifying Difficulty Scoring

Edit `backend/app/ml/difficulty.py`:

```python
def calculate_difficulty(snippet: str) -> float:
    """
    Adjust weights and factors to change how difficulty is calculated.
    """
    # Modify scoring logic
    pass
```

Then rebuild the FAISS index:
```bash
python scripts/build_faiss_index.py
```

### Fine-tuning User Encoder

Modify `backend/app/ml/user_encoder.py`:

```python
class UserEncoder:
    def encode(self, user_state: UserState) -> np.ndarray:
        """
        Adjust feature normalization and weighting.
        """
        pass
```

### Adding New Features to User State

1. **Update `UserState` schema** in `backend/app/models/schema.py`
2. **Modify `UserEncoder`** to process new features
3. **Update frontend** to collect and send new metrics

## Debugging

### Backend Debugging

**Using logging:**
```python
import logging
logger = logging.getLogger(__name__)

logger.info("User state received", extra={"wpm": 50, "accuracy": 0.95})
logger.warning("Low accuracy detected", extra={"accuracy": 0.7})
logger.error("Failed to retrieve snippet", exc_info=True)
```

**Using debugger (pdb):**
```python
import pdb; pdb.set_trace()
```

### Frontend Debugging

Use browser DevTools (F12):
- Check Network tab for API calls
- Use Console for errors
- Use React DevTools extension for component inspection

## Performance Optimization

### Backend Optimization

1. **Database Queries:** Use indexes on frequently queried columns
2. **Caching:** Cache FAISS index and snippet metadata in memory
3. **Async Operations:** Use `async/await` for I/O operations

### Frontend Optimization

1. **Code Splitting:** Use React.lazy for large components
2. **Memoization:** Use `useMemo` and `useCallback` for expensive operations
3. **Bundle Size:** Monitor with `npm run build` and analyze with webpack-bundle-analyzer

## Deployment Checklist

Before deploying:

- [ ] Run tests locally
- [ ] Update version number in `app/main.py`
- [ ] Update `CHANGELOG.md` (if exists)
- [ ] Verify `.env.example` has all required variables
- [ ] Test in Docker locally:
  ```bash
  docker build -t flowtype-backend .
  docker run -p 8000:8000 flowtype-backend
  ```
- [ ] Ensure frontend build succeeds: `npm run build`
- [ ] Check for console errors and warnings
- [ ] Verify API endpoints work with frontend

## Common Issues & Solutions

### Issue: FAISS Index Mismatch

**Problem:** Application crashes on startup with index dimension mismatch

**Solution:** Rebuild index with current embeddings:
```bash
python scripts/build_faiss_index.py
```

### Issue: Database Connection Error

**Problem:** `psycopg2.OperationalError: could not connect to server`

**Solution:** 
1. Verify `DATABASE_URL` in `.env`
2. Ensure PostgreSQL service is running
3. Check network connectivity to database server

### Issue: CORS Error on Frontend

**Problem:** API requests from frontend fail with CORS error

**Solution:**
1. Update `CORS_ORIGINS` in backend `.env`:
   ```
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000
   ```
2. Restart backend server
3. Check CORS middleware is registered in `app/main.py`

### Issue: Slow API Responses

**Problem:** API endpoints take > 1 second to respond

**Solution:**
1. Profile with logging to identify bottleneck
2. Check database query performance
3. Verify FAISS index is loaded in memory
4. Consider caching or reducing search space (k parameter)

## Contributing

1. **Create a feature branch:** `git checkout -b feature/new-feature`
2. **Make changes** following code standards
3. **Test locally** before committing
4. **Write descriptive commit messages**
5. **Push and create pull request** (if applicable)

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

