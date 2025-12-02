# ------------------------------------------------------
# schema.py  (rewritten for GRU + RL + Two-Tower ranker)
# ------------------------------------------------------
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np


# ------------------------------------------------------
# Keystroke Event (frontend → backend)
# ------------------------------------------------------
class KeystrokeEvent(BaseModel):
    timestamp: int
    key: str
    isBackspace: bool
    isCorrect: bool


# ------------------------------------------------------
# User State (frontend → backend)
# Encodes latest known rolling metrics + GRU hidden state.
# ------------------------------------------------------
class UserState(BaseModel):
    user_id: Optional[str] = None
    rollingWpm: float = 0.0
    rollingAccuracy: float = 1.0
    backspaceRate: float = 0.0
    hesitationCount: int = 0
    currentDifficulty: float = 5.0

    # Long-term stats (populated from DB or accumulated)
    overallAvgWpm: Optional[float] = 0.0
    overallBestWpm: Optional[float] = 0.0
    overallAvgAccuracy: Optional[float] = 0.0
    totalSessions: Optional[int] = 0

    # GRU hidden state (1, H)
    hiddenState: Optional[List[float]] = None

    # Recent keystrokes (last ~50)
    recentKeystrokes: Optional[List[KeystrokeEvent]] = None
    
    # History for filtering
    recentSnippetIds: Optional[List[str]] = None


# ------------------------------------------------------
# Snippet metadata during a session
# ------------------------------------------------------
class SnippetResult(BaseModel):
    snippet_id: str
    wpm: float
    accuracy: float
    difficulty: float
    started_at: Optional[int] = None
    completed_at: Optional[int] = None


# ------------------------------------------------------
# Session Create Request (frontend → backend)
# ------------------------------------------------------
class SessionCreateRequest(BaseModel):
    user_id: Optional[str]
    durationSeconds: float
    wordsTyped: int
    keystrokeData: List[KeystrokeEvent]
    wpm: float
    accuracy: float
    errors: int
    difficultyLevel: float

    # Full snippet log
    snippets: List[SnippetResult]

    # User tower state at session start
    user_state: UserState
    flowScore: Optional[float] = 0.0


# ------------------------------------------------------
# State returned after session → used for next snippet request
# ------------------------------------------------------
class UserStateUpdate(BaseModel):
    hiddenState: List[float]
    rollingWpm: float
    rollingAccuracy: float
    backspaceRate: float
    hesitationCount: int
    currentDifficulty: float


# ------------------------------------------------------
# Final API Session Response
# ------------------------------------------------------
class SessionResponse(BaseModel):
    session_id: str
    reward: float
    next_state: UserStateUpdate


# ------------------------------------------------------
# Snippet Retrieval
# ------------------------------------------------------
class SnippetRetrieveRequest(BaseModel):
    user_state: UserState
    current_snippet_id: Optional[str] = None

class SnippetResponse(BaseModel):
    snippet: Optional[dict]
    wpm_windows: dict

# ------------------------------------------------------
# User Stats (simple reporting)
# ------------------------------------------------------
class UserStats(BaseModel):
    total_sessions: int
    avg_wpm: float
    avg_accuracy: float


