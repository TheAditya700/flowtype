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
    keyup_timestamp: Optional[int] = None
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

    # Recent keystrokes (last ~50)
    recentKeystrokes: Optional[List[KeystrokeEvent]] = None
    
    # History for filtering
    recentSnippetIds: Optional[List[str]] = None
    
    # Timestamps for WPM calculation
    keystroke_timestamps: Optional[List[float]] = None


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
# Final API Session Response
# ------------------------------------------------------
class SessionResponse(BaseModel):
    session_id: str
    reward: float


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
    total_time_typing: Optional[float] = 0.0
    best_wpm_15: Optional[float] = 0.0
    best_wpm_30: Optional[float] = 0.0
    best_wpm_60: Optional[float] = 0.0
    best_wpm_120: Optional[float] = 0.0

class UserProfile(BaseModel):
    user_id: str
    username: Optional[str] = None # Added username
    features: dict
    stats: UserStats

# ------------------------------------------------------
# Authentication Schemas
# ------------------------------------------------------
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel): # Used for JWT payload validation
    username: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    username: str

    class Config:
        from_attributes = True


# ------------------------------------------------------
# Analytics Schemas
# ------------------------------------------------------
class SnippetBoundary(BaseModel):
    startTime: int
    endTime: int

class AnalyticsRequest(BaseModel):
    keystrokeData: List[KeystrokeEvent]
    wpm: float
    accuracy: float
    snippetBoundaries: Optional[List[SnippetBoundary]] = None

class SpeedPoint(BaseModel):
    time: float
    wpm: float
    rawWpm: float
    errors: int

class ReplayEvent(BaseModel):
    char: str
    iki: float
    isChunkStart: bool
    isError: bool
    snippetIndex: Optional[int] = None
    isRollover: Optional[bool] = False

class AnalyticsResponse(BaseModel):
    smoothness: float
    rollover: float
    leftFluency: float
    rightFluency: float
    crossFluency: float
    speed: float
    accuracy: float
    
    # Detailed stats for widgets
    avgIki: float
    kspc: float
    errors: int
    heatmapData: dict[str, dict[str, float]]
    avgChunkLength: float
    
    # Time Series and Replay
    speedSeries: List[SpeedPoint]
    replayEvents: List[ReplayEvent]
