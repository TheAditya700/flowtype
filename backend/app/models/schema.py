from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid

# Keystroke and User State models from frontend types
class KeystrokeEvent(BaseModel):
    timestamp: int
    key: str
    isBackspace: bool
    isCorrect: bool

class UserState(BaseModel):
    rollingWpm: float
    rollingAccuracy: float
    backspaceRate: float
    hesitationCount: int
    recentErrors: List[str]
    currentDifficulty: float
    recentSnippetIds: Optional[List[str]] = None  # Track recently shown snippets to avoid repeats

# API Models
class SnippetRetrieveRequest(BaseModel):
    user_state: UserState
    current_snippet_id: Optional[str] = None  # The snippet currently being typed to avoid returning the same one

class SnippetResponse(BaseModel):
    id: uuid.UUID
    words: str
    difficulty: float

    class Config:
        orm_mode = True

class SnippetLog(BaseModel):
    snippet_id: uuid.UUID
    started_at: datetime
    completed_at: datetime
    wpm: float
    accuracy: float
    difficulty: float

class SessionCreateRequest(BaseModel):
    startedAt: datetime
    durationSeconds: float
    wordsTyped: int
    errors: int
    wpm: float
    accuracy: float
    difficultyLevel: float
    keystrokeData: List[KeystrokeEvent]
    user_id: Optional[uuid.UUID] = None
    snippets: List[SnippetLog] = []

class SessionResponse(BaseModel):
    id: uuid.UUID
    flow_score: Optional[float] = None

    class Config:
        orm_mode = True

class UserStats(BaseModel):
    total_sessions: int
    avg_wpm: float
    avg_accuracy: float
    # Add more stats as needed

    class Config:
        orm_mode = True
