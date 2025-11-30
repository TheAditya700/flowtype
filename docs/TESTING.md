# FlowType Testing & Quality Assurance Guide

This document outlines testing strategies and quality assurance practices for FlowType.

## Testing Strategy

FlowType uses a layered testing approach:
1. **Unit Tests:** Individual function and module testing
2. **Integration Tests:** API endpoint and database interaction testing
3. **End-to-End Tests:** Full user workflows
4. **Performance Tests:** Load testing and latency measurements

## Backend Testing

### Unit Tests

**Location:** `backend/tests/`

**Test Difficulty Scoring:**
```python
# tests/test_difficulty.py
import pytest
from app.ml.difficulty import calculate_difficulty

def test_calculate_difficulty_word_length():
    """Longer words increase difficulty."""
    easy = calculate_difficulty("cat dog bird")
    hard = calculate_difficulty("encyclopedia pseudonym philosophy")
    assert hard > easy

def test_calculate_difficulty_punctuation():
    """Punctuation increases difficulty."""
    no_punct = calculate_difficulty("the quick brown fox")
    with_punct = calculate_difficulty("the quick brown-fox!")
    assert with_punct > no_punct

def test_calculate_difficulty_rare_letters():
    """Rare letters increase difficulty."""
    common = calculate_difficulty("the dog sat")
    rare = calculate_difficulty("the aquatic zebra")
    assert rare > common
```

**Test Metric Calculations:**
```python
# tests/test_metrics.py
import pytest
from app.utils.metrics import calculate_wpm, calculate_accuracy

def test_calculate_wpm():
    """Test WPM calculation."""
    assert calculate_wpm(50, 60) == 50.0  # 50 words in 60 seconds
    assert calculate_wpm(100, 30) == 200.0  # 100 words in 30 seconds
    assert calculate_wpm(0, 60) == 0.0  # Zero words

def test_calculate_accuracy():
    """Test accuracy calculation."""
    assert calculate_accuracy(95, 100) == 0.95
    assert calculate_accuracy(100, 100) == 1.0
    assert calculate_accuracy(0, 100) == 0.0

def test_accuracy_zero_division():
    """Handle zero total characters."""
    assert calculate_accuracy(0, 0) == 0.0
```

**Test User Encoding:**
```python
# tests/test_user_encoder.py
import pytest
import numpy as np
from app.ml.user_encoder import UserEncoder
from app.models.schema import UserState

def test_user_encoder_output_shape():
    """User encoder produces correct embedding size."""
    encoder = UserEncoder()
    user_state = UserState(
        rollingWpm=50.0,
        rollingAccuracy=0.95,
        backspaceRate=0.1,
        hesitationCount=2,
        recentErrors=[],
        currentDifficulty=3.0
    )
    embedding = encoder.encode(user_state)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)

def test_user_encoder_normalization():
    """Embeddings are properly normalized."""
    encoder = UserEncoder()
    user_state = UserState(
        rollingWpm=100.0,
        rollingAccuracy=1.0,
        backspaceRate=0.0,
        hesitationCount=0,
        recentErrors=[],
        currentDifficulty=5.0
    )
    embedding = encoder.encode(user_state)
    # Check that values are in reasonable range
    assert np.all(np.isfinite(embedding))
    assert np.max(np.abs(embedding)) <= 10.0
```

### Integration Tests

**Testing API Endpoints:**
```python
# tests/test_snippets.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_retrieve_snippets_endpoint():
    """Test snippet retrieval endpoint."""
    response = client.post("/api/snippets/retrieve", json={
        "user_state": {
            "rollingWpm": 50.0,
            "rollingAccuracy": 0.95,
            "backspaceRate": 0.1,
            "hesitationCount": 2,
            "recentErrors": [],
            "currentDifficulty": 3.0
        }
    })
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "id" in data[0]
    assert "words" in data[0]
    assert "difficulty" in data[0]

def test_retrieve_snippets_missing_fields():
    """Test error handling for incomplete requests."""
    response = client.post("/api/snippets/retrieve", json={
        "user_state": {}  # Missing required fields
    })
    assert response.status_code == 422  # Validation error

def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

**Testing Session Creation:**
```python
# tests/test_sessions.py
def test_create_session():
    """Test session creation and storage."""
    session_data = {
        "startedAt": "2023-10-27T10:00:00Z",
        "durationSeconds": 60.5,
        "wordsTyped": 50,
        "errors": 2,
        "wpm": 49.5,
        "accuracy": 0.98,
        "difficultyLevel": 3.2,
        "keystrokeData": [],
        "user_id": "test-user"
    }
    response = client.post("/api/sessions", json=session_data)
    assert response.status_code == 200
    result = response.json()
    assert "id" in result
    assert "flow_score" in result
```

### Running Tests

```bash
cd backend

# Run all tests
pytest

# Run specific test file
pytest tests/test_difficulty.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=app tests/

# Run only unit tests
pytest tests/ -m "not integration"
```

### Test Configuration

Add to `backend/pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    integration: marks tests as integration tests
    slow: marks tests as slow
```

## Frontend Testing

### Component Testing with Vitest/Jest

```typescript
// src/components/__tests__/TypingZone.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import TypingZone from '../TypingZone';

describe('TypingZone', () => {
  it('renders typing input', () => {
    render(<TypingZone snippet="hello world" onComplete={() => {}} />);
    const input = screen.getByRole('textbox');
    expect(input).toBeInTheDocument();
  });

  it('updates input value on keystroke', async () => {
    const { user } = render(
      <TypingZone snippet="hello" onComplete={() => {}} />
    );
    const input = screen.getByRole('textbox');
    await user.type(input, 'h');
    expect(input).toHaveValue('h');
  });
});
```

### Hook Testing

```typescript
// src/hooks/__tests__/useWPMCalculation.test.ts
import { renderHook, act } from '@testing-library/react';
import { useWPMCalculation } from '../useWPMCalculation';

describe('useWPMCalculation', () => {
  it('calculates correct WPM', () => {
    const { result } = renderHook(() => useWPMCalculation());
    
    act(() => {
      result.current.updateMetrics({
        wordsTyped: 50,
        elapsedSeconds: 60
      });
    });
    
    expect(result.current.wpm).toBe(50);
  });
});
```

### Running Frontend Tests

```bash
cd frontend

# Run tests
npm run test

# Run tests in watch mode
npm run test -- --watch

# Generate coverage report
npm run test -- --coverage
```

## Performance Testing

### Backend Performance Tests

```python
# tests/test_performance.py
import pytest
import time
from app.ml.vector_store import VectorStore

@pytest.mark.slow
def test_faiss_search_latency():
    """Ensure FAISS search is fast enough."""
    vector_store = VectorStore()
    user_embedding = np.random.rand(384)
    
    start = time.time()
    ids, distances = vector_store.search(user_embedding, k=20)
    elapsed = time.time() - start
    
    assert elapsed < 0.01  # Must be under 10ms
    assert len(ids) == 20

@pytest.mark.slow
def test_endpoint_response_time():
    """Ensure API endpoint responds quickly."""
    client = TestClient(app)
    
    start = time.time()
    response = client.post("/api/snippets/retrieve", json={...})
    elapsed = time.time() - start
    
    assert response.status_code == 200
    assert elapsed < 0.05  # Must be under 50ms
```

## Load Testing

Use Apache JMeter or similar tools:

```bash
# Install JMeter
brew install jmeter

# Run load test (example)
jmeter -n -t test_plan.jmx -l results.jtl
```

**Test Scenario:**
- 100 concurrent users
- 50 requests each
- Ramp-up over 30 seconds
- Measure: avg response time, 95th percentile, errors

## Code Quality

### Linting and Formatting

**Backend (Python):**
```bash
cd backend

# Format code with black
black .

# Lint with pylint
pylint app/

# Type checking with mypy
mypy app/
```

**Frontend (TypeScript):**
```bash
cd frontend

# Format code with Prettier
npm run format

# Lint with ESLint
npm run lint

# Type checking (built into TypeScript)
npm run type-check
```

### Code Coverage Goals

- **Backend:** Minimum 70% coverage for critical modules
  - `app/ml/` : 85%
  - `app/models/` : 80%
  - `app/routers/` : 75%

- **Frontend:** Minimum 60% coverage
  - `components/` : 75%
  - `hooks/` : 80%
  - `utils/` : 70%

Check coverage:
```bash
# Backend
cd backend
pytest --cov=app --cov-report=html

# Frontend
cd frontend
npm run test -- --coverage
```

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt pytest pytest-cov
      - name: Run tests
        run: |
          cd backend
          pytest --cov=app
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: |
          cd frontend
          npm install
      - name: Run tests
        run: |
          cd frontend
          npm run test
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Manual Testing Checklist

### User Workflow Testing

- [ ] User creates account (if applicable)
- [ ] User starts typing session
- [ ] Real-time WPM updates correctly
- [ ] Accuracy calculates correctly
- [ ] Difficulty adjusts based on performance
- [ ] Session completes and saves
- [ ] User can view session history
- [ ] Stats display correctly

### Edge Case Testing

- [ ] Very fast typing (> 150 WPM)
- [ ] Very slow typing (< 10 WPM)
- [ ] High error rate (> 10%)
- [ ] Perfect accuracy (0% errors)
- [ ] Long sessions (> 30 minutes)
- [ ] Browser refresh during session
- [ ] Network interruption and recovery
- [ ] Mobile/tablet responsiveness

## Debugging Failed Tests

### Backend Test Failures

```bash
# Run with verbose output and stop on first failure
pytest -vvx

# Run with debugging info
pytest --pdb

# Run specific test
pytest tests/test_file.py::TestClass::test_method -vv
```

### Frontend Test Failures

```bash
# Watch mode for iterative debugging
npm run test -- --watch

# Debug in browser
npm run test -- --inspect-brk

# Use --debug flag with specific test file
npm run test -- --debug src/components/__tests__/Component.test.tsx
```

## Performance Profiling

### Backend Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
vector_store.search(embedding, k=20)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Frontend Profiling

Use React DevTools Profiler:
1. Open React DevTools in browser
2. Click "Profiler" tab
3. Record performance session
4. Analyze component render times

