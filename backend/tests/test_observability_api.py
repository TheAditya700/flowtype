"""
Observability API Integration Tests.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from app.database import Base
from app.main import app
from app.models.db_models import TypingSession, ModelSnapshots, User


# Create in-memory SQLite database for tests
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Dependency override for database."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def test_client():
    """Create test client with fresh database for each test."""
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Mock vector store
    mock_vector_store = MagicMock()
    app.state.vector_store = mock_vector_store

    # Override database dependency
    from app.routers import observability

    app.dependency_overrides[observability.get_db] = override_get_db

    client = TestClient(app)
    yield client

    # Cleanup
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


@pytest.fixture
def sample_data(test_client):
    """Populate database with sample data."""
    db = TestingSessionLocal()

    # Create a user
    user = User(
        username="obs_user",
        hashed_password="hashed_pw",
        features={"ema": {"ema_mean": [0.0] * 30}},  # Minimal features
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Create model snapshots
    for i in range(5):
        snap = ModelSnapshots(
            model_version=f"v0.0.{i}",
            mean_precision=0.8 + (i * 0.01),
            median_precision=0.8,
            p90_precision=0.9,
            p99_precision=0.99,
            mean_variance=0.1,
            fraction_high_confidence=0.5,
            mean_abs_weight=0.1,
            p90_abs_weight=0.2,
            fraction_near_zero_mean=0.1,
            fraction_confident_irrelevant=0.05,
            mean_abs_delta_mean=0.05,
            mean_delta_precision=0.01,
            fraction_weights_updated=0.1,
            top_importance_weights=["w1", "w2"],
            top_certain_weights=["c1", "c2"],
            top_uncertain_weights=["u1", "u2"],
            created_at=datetime.now(timezone.utc) - timedelta(hours=5 - i),
        )
        db.add(snap)

    # Create typing sessions
    for i in range(10):
        session = TypingSession(
            user_id=user.id,
            duration_seconds=60.0,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10 * i),
            reward=1.0,
            actual_accuracy=0.95,
            actual_wpm=60.0,
            actual_consistency=90.0,
            errors=0,
            raw_wpm=60.0,
            snippet_ids=["s1"],
        )
        db.add(session)

    db.commit()
    db.close()


def test_observability_header(test_client, sample_data):
    """Test header KPIs."""
    response = test_client.get("/api/observability/header")
    assert response.status_code == 200
    data = response.json()
    assert data["total_sessions"] == 10
    assert data["model_version"] == "v0.0.4"
    assert "active_users" in data


def test_learning_health(test_client, sample_data):
    """Test learning health endpoint."""
    response = test_client.get("/api/observability/learning_health?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    assert len(data["points"]) > 0
    assert "mean_precision" in data["points"][0]


def test_user_skills(test_client, sample_data):
    """Test user skills endpoint (was user_skills_all)."""
    # Note: test_observability.py used /user_skills, but actual router has /user_skills_all
    response = test_client.get("/api/observability/user_skills_all?top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert "impact" in data
    assert "certain" in data
    assert "uncertain" in data
    assert len(data["impact"]) <= 3


def test_agent_effectiveness(test_client, sample_data):
    """Test agent effectiveness endpoint."""
    response = test_client.get("/api/observability/agent_effectiveness")
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    assert len(data["points"]) > 0
    assert "mean_reward" in data["points"][0]


def test_learning_activity(test_client, sample_data):
    """Test learning activity endpoint."""
    response = test_client.get("/api/observability/learning_activity")
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    assert len(data["points"]) > 0
    assert "fraction_weights_updated" in data["points"][0]
