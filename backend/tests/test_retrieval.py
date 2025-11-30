import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.schema import UserState

client = TestClient(app)

def test_retrieve_endpoint():
    """
    Test the /api/snippets/retrieve endpoint.
    NOTE: This test will fail until the vector store is properly
    integrated and loaded during app startup for the test environment.
    """
    user_state = UserState(
        rollingWpm=50.0,
        rollingAccuracy=0.95,
        backspaceRate=0.1,
        hesitationCount=2,
        recentErrors=[],
        currentDifficulty=3.0
    )
    
    response = client.post("/api/snippets/retrieve", json={"user_state": user_state.dict()})
    
    # This will likely fail with 404 until the app state is managed in tests
    assert response.status_code == 200 
    assert isinstance(response.json(), list)
    # In our placeholder, we return 2 snippets
    assert len(response.json()) == 2
    assert "words" in response.json()[0]
