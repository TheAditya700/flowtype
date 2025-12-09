import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Index, JSON, BigInteger, func, Boolean
from sqlalchemy.sql import func
from app.database import Base
from sqlalchemy.orm import relationship

# ------------------------------------------------------
# db_models.py (rewritten for Stateful GRU + RL)
# ------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True, nullable=True) # Making nullable for now to support anonymous if needed, will make not null later after user migration
    hashed_password = Column(String, nullable=True) # Making nullable for now
    created_at = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now(), onupdate=func.now())

    # Best WPM stats for various intervals (JSON: {"15": wpm, "30": wpm, ...})
    best_wpms = Column(JSON, default={"15": 0.0, "30": 0.0, "60": 0.0, "120": 0.0})

    # Long-term feature storage (serialized UserFeatureExtractor + Agent EMA)
    features = Column(JSON, default={})


class Snippet(Base):
    __tablename__ = "snippets"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Raw snippet text
    text = Column(String, nullable=False)

    # Tokenized words
    words = Column(JSON, nullable=False)

    # Metadata
    word_count = Column(Integer, nullable=False)

    # Full difficulty feature vector (raw)
    features = Column(JSON, nullable=False)

    # Normalized feature vector (post-scaling)
    normalized_features = Column(JSON, nullable=True)

    # Final embedding vector (e.g., PCA/UMAP/MLP output, float list)
    embedding = Column(JSON, nullable=True)

    # Output of Snippet Tower MLP (used for search)
    processed_embedding = Column(JSON, nullable=True)

    # Difficulty score (optional model-generated scalar)
    difficulty_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_difficulty', 'difficulty_score'),
    )



# -----------------------------------------
# Typing Session
# -----------------------------------------
class TypingSession(Base):
    __tablename__ = "typing_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True, index=True)

    duration_seconds = Column(Float)
    words_typed = Column(Integer)
    errors = Column(Integer)
    backspaces = Column(Integer)

    final_wpm = Column(Float)
    accuracy = Column(Float)

    starting_difficulty = Column(Float)
    ending_difficulty = Column(Float)
    avg_difficulty = Column(Float)

    flow_score = Column(Float)

    # RL reward
    reward = Column(Float, nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    # relationships
    keystrokes = relationship("KeystrokeEventDB", back_populates="session")
    snippet_usages = relationship("SnippetUsage", back_populates="session")


# -----------------------------------------
# Per-Keystroke Storage (for GRU training)
# -----------------------------------------
class KeystrokeEventDB(Base):
    __tablename__ = "keystroke_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("typing_sessions.id"))
    timestamp = Column(BigInteger)
    key = Column(String)
    is_backspace = Column(Boolean)
    is_correct = Column(Boolean)

    session = relationship("TypingSession", back_populates="keystrokes")


# -----------------------------------------
# Per-Snippet Usage Metadata (for ranking/RL)
# -----------------------------------------
class SnippetUsage(Base):
    __tablename__ = "snippet_usage"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("typing_sessions.id"), index=True)
    snippet_id = Column(String, index=True)

    user_wpm = Column(Float)
    user_accuracy = Column(Float)
    snippet_position = Column(Integer)

    difficulty_snapshot = Column(Float)

    created_at = Column(DateTime, server_default=func.now())

    session = relationship("TypingSession", back_populates="snippet_usages")
