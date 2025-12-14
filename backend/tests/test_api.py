"""
API Integration Tests using FastAPI TestClient.

Covers:
- Health endpoint
- Auth endpoints (register, login, token validation)
- User endpoints
- Core request/response contracts

Uses an in-memory SQLite database and mock vector store for isolation.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock, patch
import numpy as np

from app.database import Base
from app.main import app
from app.database import SessionLocal


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
    mock_vector_store.search.return_value = [
        {"id": "snippet-1", "words": "test words here", "difficulty": 0.5}
    ]

    app.state.vector_store = mock_vector_store

    # Override database dependency
    from app.routers import auth, snippets, sessions, users

    for router_module in [auth, snippets, sessions, users]:
        if hasattr(router_module, "get_db"):
            app.dependency_overrides[router_module.get_db] = override_get_db

    client = TestClient(app)
    yield client

    # Cleanup
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_ok(self, test_client):
        """Health endpoint should return status ok."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_api_info(self, test_client):
        """Root should return API info."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "FlowType" in data["message"]
        assert "version" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_register_new_user(self, test_client):
        """Should register a new user successfully."""
        response = test_client.post(
            "/api/auth/register",
            json={"username": "testuser", "password": "testpass123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert "id" in data
        # Password should NOT be in response
        assert "password" not in data
        assert "hashed_password" not in data

    def test_register_duplicate_username(self, test_client):
        """Should reject duplicate username."""
        # Register first user
        test_client.post(
            "/api/auth/register",
            json={"username": "duplicate", "password": "pass123"},
        )

        # Try to register same username
        response = test_client.post(
            "/api/auth/register",
            json={"username": "duplicate", "password": "different"},
        )

        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    def test_login_success(self, test_client):
        """Should login and return access token."""
        # Register
        test_client.post(
            "/api/auth/register",
            json={"username": "logintest", "password": "mypassword"},
        )

        # Login
        response = test_client.post(
            "/api/auth/token",
            data={"username": "logintest", "password": "mypassword"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, test_client):
        """Should reject wrong password."""
        # Register
        test_client.post(
            "/api/auth/register",
            json={"username": "wrongpass", "password": "correct"},
        )

        # Login with wrong password
        response = test_client.post(
            "/api/auth/token",
            data={"username": "wrongpass", "password": "incorrect"},
        )

        assert response.status_code == 401
        assert "Incorrect" in response.json()["detail"]

    def test_login_nonexistent_user(self, test_client):
        """Should reject non-existent user."""
        response = test_client.post(
            "/api/auth/token",
            data={"username": "nouser", "password": "nopass"},
        )

        assert response.status_code == 401

    def test_get_current_user(self, test_client):
        """Should return current user with valid token."""
        # Register and login
        test_client.post(
            "/api/auth/register",
            json={"username": "meuser", "password": "mepass"},
        )
        login_response = test_client.post(
            "/api/auth/token",
            data={"username": "meuser", "password": "mepass"},
        )
        token = login_response.json()["access_token"]

        # Get current user
        response = test_client.get(
            "/api/auth/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["username"] == "meuser"

    def test_get_current_user_invalid_token(self, test_client):
        """Should reject invalid token."""
        response = test_client.get(
            "/api/auth/users/me",
            headers={"Authorization": "Bearer invalidtoken"},
        )

        assert response.status_code == 401

    def test_get_current_user_no_token(self, test_client):
        """Should reject missing token."""
        response = test_client.get("/api/auth/users/me")

        assert response.status_code == 401


class TestUsernameChange:
    """Tests for username change endpoint."""

    def _get_auth_token(self, client, username, password):
        """Helper to register and get token."""
        client.post(
            "/api/auth/register",
            json={"username": username, "password": password},
        )
        response = client.post(
            "/api/auth/token",
            data={"username": username, "password": password},
        )
        return response.json()["access_token"]

    def test_change_username_success(self, test_client):
        """Should change username successfully."""
        token = self._get_auth_token(test_client, "oldname", "password")

        response = test_client.put(
            "/api/auth/users/change-username",
            json={"new_username": "newname"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["username"] == "newname"

    def test_change_username_to_taken(self, test_client):
        """Should reject changing to taken username."""
        # Create two users
        token1 = self._get_auth_token(test_client, "user1", "pass1")
        self._get_auth_token(test_client, "user2", "pass2")

        # Try to change user1 to user2's name
        response = test_client.put(
            "/api/auth/users/change-username",
            json={"new_username": "user2"},
            headers={"Authorization": f"Bearer {token1}"},
        )

        assert response.status_code == 400
        assert "already taken" in response.json()["detail"]

    def test_change_username_empty(self, test_client):
        """Should reject empty username."""
        token = self._get_auth_token(test_client, "emptytest", "pass")

        response = test_client.put(
            "/api/auth/users/change-username",
            json={"new_username": ""},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 400


class TestPasswordChange:
    """Tests for password change endpoint."""

    def _get_auth_token(self, client, username, password):
        client.post(
            "/api/auth/register",
            json={"username": username, "password": password},
        )
        response = client.post(
            "/api/auth/token",
            data={"username": username, "password": password},
        )
        return response.json()["access_token"]

    def test_change_password_success(self, test_client):
        """Should change password successfully."""
        token = self._get_auth_token(test_client, "passchange", "oldpass")

        response = test_client.put(
            "/api/auth/users/change-password",
            json={"current_password": "oldpass", "new_password": "newpass"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert "successfully" in response.json()["message"]

        # Verify new password works
        login_response = test_client.post(
            "/api/auth/token",
            data={"username": "passchange", "password": "newpass"},
        )
        assert login_response.status_code == 200

    def test_change_password_wrong_current(self, test_client):
        """Should reject wrong current password."""
        token = self._get_auth_token(test_client, "wrongcurrent", "correct")

        response = test_client.put(
            "/api/auth/users/change-password",
            json={"current_password": "incorrect", "new_password": "newpass"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 401


class TestAccountDeletion:
    """Tests for account deletion endpoint."""

    def _get_auth_token(self, client, username, password):
        client.post(
            "/api/auth/register",
            json={"username": username, "password": password},
        )
        response = client.post(
            "/api/auth/token",
            data={"username": username, "password": password},
        )
        return response.json()["access_token"]

    def test_delete_account_success(self, test_client):
        """Should delete account successfully."""
        token = self._get_auth_token(test_client, "todelete", "deletepass")

        response = test_client.request(
            "DELETE",
            "/api/auth/users/delete-account",
            json={"password": "deletepass"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert "deleted" in response.json()["message"]

        # Verify can't login anymore
        login_response = test_client.post(
            "/api/auth/token",
            data={"username": "todelete", "password": "deletepass"},
        )
        assert login_response.status_code == 401

    def test_delete_account_wrong_password(self, test_client):
        """Should reject wrong password for deletion."""
        token = self._get_auth_token(test_client, "nodelete", "rightpass")

        response = test_client.request(
            "DELETE",
            "/api/auth/users/delete-account",
            json={"password": "wrongpass"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 401


class TestSnippetRetrieval:
    """Tests for snippet retrieval endpoint."""

    def test_retrieve_snippets_basic(self, test_client):
        """Should retrieve snippets with minimal request."""
        response = test_client.post(
            "/api/snippets/retrieve",
            json={
                "user_state": {},
                "current_snippet_id": None,
            },
        )

        # Should succeed (with mock vector store)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data or "words" in data or "snippet" in data

    def test_retrieve_snippets_with_user_state(self, test_client):
        """Should handle user state in request."""
        response = test_client.post(
            "/api/snippets/retrieve",
            json={
                "user_state": {
                    "user_id": "test-user-123",
                    "recentSnippetIds": ["old-1", "old-2"],
                    "rollingWpm": 55.0,
                },
                "current_snippet_id": "prev-snippet",
            },
        )

        assert response.status_code == 200


class TestAPIContracts:
    """Tests verifying API request/response contracts."""

    def test_register_requires_username_and_password(self, test_client):
        """Register should require both fields."""
        # Missing password
        response = test_client.post(
            "/api/auth/register",
            json={"username": "onlyuser"},
        )
        assert response.status_code == 422  # Validation error

        # Missing username
        response = test_client.post(
            "/api/auth/register",
            json={"password": "onlypass"},
        )
        assert response.status_code == 422

    def test_token_response_format(self, test_client):
        """Token response should match OAuth2 format."""
        test_client.post(
            "/api/auth/register",
            json={"username": "tokenformat", "password": "pass"},
        )

        response = test_client.post(
            "/api/auth/token",
            data={"username": "tokenformat", "password": "pass"},
        )

        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

    def test_user_response_format(self, test_client):
        """User response should match UserResponse schema."""
        test_client.post(
            "/api/auth/register",
            json={"username": "userformat", "password": "pass"},
        )
        login = test_client.post(
            "/api/auth/token",
            data={"username": "userformat", "password": "pass"},
        )
        token = login.json()["access_token"]

        response = test_client.get(
            "/api/auth/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        data = response.json()
        assert "id" in data
        assert "username" in data
        # Sensitive fields should not be exposed
        assert "password" not in data
        assert "hashed_password" not in data
