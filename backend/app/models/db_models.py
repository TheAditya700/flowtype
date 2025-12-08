import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Index, JSON, TypeDecorator, DateTime, ForeignKey, func, Boolean
from sqlalchemy.sql import func
from app.database import Base
from sqlalchemy.orm import relationship

# ------------------------------------------------------
# db_models.py (rewritten for Stateful GRU + Telemetry)
# ------------------------------------------------------

class GUID(TypeDecorator):
    """Platform-independent GUID type that uses CHAR(32) on SQLite and UUID on PostgreSQL."""
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(value) if value else None

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(value)
        except (ValueError, AttributeError):
            # If it's not a valid UUID, return as-is (e.g., user-001, sess-001)
            return value

class User(Base):
    __tablename__ = "users"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, index=True, nullable=True) # Making nullable for now to support anonymous if needed, will make not null later after user migration
    hashed_password = Column(String, nullable=True) # Making nullable for now
    created_at = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now(), onupdate=func.now())

    # Long-term feature storage (serialized UserFeatureExtractor)
    features = Column(JSON, default={})


class Snippet(Base):
    __tablename__ = "snippets"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

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

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
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

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    session_id = Column(String, ForeignKey("typing_sessions.id"))
    timestamp = Column(Integer)
    key = Column(String)
    is_backspace = Column(Boolean)
    is_correct = Column(Boolean)

    session = relationship("TypingSession", back_populates="keystrokes")


# -----------------------------------------
# Per-Snippet Usage Metadata (for ranking/RL)
# -----------------------------------------
class SnippetUsage(Base):
    __tablename__ = "snippet_usage"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("typing_sessions.id"), index=True)
    snippet_id = Column(String, index=True)

    user_wpm = Column(Float)
    user_accuracy = Column(Float)
    snippet_position = Column(Integer)

    difficulty_snapshot = Column(Float)

    created_at = Column(DateTime, server_default=func.now())

    session = relationship("TypingSession", back_populates="snippet_usages")


# -----------------------------------------
# Telemetry Storage (raw logs for training)
# -----------------------------------------
class TelemetrySnippetRaw(Base):
    __tablename__ = "telemetry_snippet_raw"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    user_id = Column(String, index=True)
    snippet_id = Column(String, index=True)
    session_id = Column(String, nullable=True)

    # the full telemetry payload (large but durable)
    payload = Column(JSON)

    # optional model fields useful for online learning
    model_score = Column(Float)
    reward_estimate = Column(Float)

    source = Column(String)
    created_at = Column(DateTime, server_default=func.now())
