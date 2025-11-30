# FlowType API Documentation

This document outlines the API endpoints for the FlowType application.

## Endpoints

### `POST /api/snippets/retrieve`

Retrieves and ranks typing snippets based on the user's current state.

**Request Body:**
```json
{
  "user_state": {
    "rollingWpm": 50.0,
    "rollingAccuracy": 0.95,
    "backspaceRate": 0.1,
    "hesitationCount": 2,
    "recentErrors": ["e", "r"],
    "currentDifficulty": 3.0
  }
}
```

**Response:**
```json
[
  {
    "id": "c1a7e7f8-7a7b-4e4f-8f0e-3b1e9b0a1b9a",
    "words": "the quick brown fox",
    "difficulty": 2.5
  },
  {
    "id": "d2b8f8g9-8b8c-5f5g-9g1f-4c2f0c1b2c0b",
    "words": "jumps over the lazy dog",
    "difficulty": 3.1
  }
]
```

### `POST /api/sessions`

Saves a completed typing session to the database.

**Request Body:**
```json
{
  "startedAt": "2023-10-27T10:00:00Z",
  "durationSeconds": 60.5,
  "wordsTyped": 50,
  "errors": 2,
  "wpm": 49.5,
  "accuracy": 0.98,
  "difficultyLevel": 3.2,
  "keystrokeData": [
    {"timestamp": 1678886400000, "key": "t", "isBackspace": false, "isCorrect": true},
    {"timestamp": 1678886400100, "key": "h", "isBackspace": false, "isCorrect": true}
  ],
  "user_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef"
}
```

**Response:**
```json
{
  "id": "unique-session-id",
  "flow_score": 123.45
}
```

### `GET /api/users/{user_id}/stats`

Retrieves aggregate typing statistics for a given user.

**Path Parameters:**
*   `user_id`: UUID of the user.

**Response:**
```json
{
  "total_sessions": 10,
  "avg_wpm": 60.2,
  "avg_accuracy": 0.97
}
```
