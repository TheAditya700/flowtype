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
    
    # Session metadata
    duration_seconds = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
    
    # User state at session time
    user_embedding = Column(JSON, nullable=True)  # 130-dim user state vector
    
    # Snippets typed (list of snippet IDs in order)
    snippet_ids = Column(JSON, nullable=False, default=[])  # ["id1", "id2", ...]
    snippet_embeddings = Column(JSON, nullable=True)  # List of 16-dim embeddings
    
    # Keystroke data (full list of keystroke events)
    keystroke_events = Column(JSON, nullable=False, default=[])
    
    # Actual performance metrics
    actual_wpm = Column(Float)
    actual_accuracy = Column(Float)
    actual_consistency = Column(Float)  # smoothness score
    
    # Predicted metrics (from LinTS agent at session start)
    predicted_wpm = Column(Float, nullable=True)
    predicted_accuracy = Column(Float, nullable=True)
    predicted_consistency = Column(Float, nullable=True)
    
    # Additional stats
    errors = Column(Integer)
    raw_wpm = Column(Float)
    
    # RL reward (for agent updates)
    reward = Column(Float, nullable=True)
