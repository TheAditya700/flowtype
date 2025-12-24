import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import (
    String,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    BigInteger,
    func,
    Boolean,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from app.database import Base


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    username: Mapped[Optional[str]] = mapped_column(
        String, unique=True, index=True, nullable=True
    )  # Nullable to support anonymous users
    hashed_password: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )  # Nullable for anonymous users
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    last_active: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Anonymous user tracking
    is_anonymous: Mapped[bool] = mapped_column(
        Boolean, default=True
    )  # True if user hasn't registered yet
    merged_into: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True
    )  # ID of authenticated user this profile was merged into

    # Best WPM stats for various intervals (JSON: {"15": wpm, "30": wpm, ...})
    best_wpms: Mapped[Dict[str, float]] = mapped_column(
        JSON, default={"15": 0.0, "30": 0.0, "60": 0.0, "120": 0.0}
    )

    # Long-term feature storage (serialized UserFeatureExtractor + Agent EMA)
    features: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})


class Snippet(Base):
    __tablename__ = "snippets"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # Raw snippet text
    text: Mapped[str] = mapped_column(String, nullable=False)

    # Tokenized words
    words: Mapped[List[str]] = mapped_column(JSON, nullable=False)

    # Metadata
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Full difficulty feature vector (raw)
    features: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Normalized feature vector (post-scaling)
    normalized_features: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON, nullable=True
    )

    # Final embedding vector (e.g., PCA/UMAP/MLP output, float list)
    embedding: Mapped[Optional[List[float]]] = mapped_column(JSON, nullable=True)

    # Output of Snippet Tower MLP (used for search)
    processed_embedding: Mapped[Optional[List[float]]] = mapped_column(
        JSON, nullable=True
    )

    # Difficulty score (optional model-generated scalar)
    difficulty_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (Index("idx_difficulty", "difficulty_score"),)


class TypingSession(Base):
    __tablename__ = "typing_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)

    # Session metadata
    duration_seconds: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # User state at session time
    user_embedding: Mapped[Optional[List[float]]] = mapped_column(
        JSON, nullable=True
    )  # 130-dim user state vector

    # Snippets typed (list of snippet IDs in order)
    snippet_ids: Mapped[List[str]] = mapped_column(
        JSON, nullable=False, default=[]
    )  # ["id1", "id2", ...]
    snippet_embeddings: Mapped[Optional[List[List[float]]]] = mapped_column(
        JSON, nullable=True
    )  # List of 16-dim embeddings

    # Keystroke data (full list of keystroke events)
    keystroke_events: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=[]
    )

    # Actual performance metrics
    actual_wpm: Mapped[float] = mapped_column(Float)
    actual_accuracy: Mapped[float] = mapped_column(Float)
    actual_consistency: Mapped[float] = mapped_column(Float)  # smoothness score

    # Predicted metrics (from LinTS agent at session start)
    predicted_wpm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_consistency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Additional stats
    errors: Mapped[int] = mapped_column(Integer)
    raw_wpm: Mapped[float] = mapped_column(Float)

    # RL reward (for agent updates)
    reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class ModelSnapshots(Base):
    __tablename__ = "model_snapshots"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    weights_uri: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )  # Points to latest weights artifact

    # Belief confidence
    mean_precision: Mapped[float] = mapped_column(Float, nullable=False)
    median_precision: Mapped[float] = mapped_column(Float, nullable=False)
    p90_precision: Mapped[float] = mapped_column(Float, nullable=False)
    p99_precision: Mapped[float] = mapped_column(Float, nullable=False)
    mean_variance: Mapped[float] = mapped_column(Float, nullable=False)
    fraction_high_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Belief structure
    mean_abs_weight: Mapped[float] = mapped_column(Float, nullable=False)
    p90_abs_weight: Mapped[float] = mapped_column(Float, nullable=False)
    fraction_near_zero_mean: Mapped[float] = mapped_column(Float, nullable=False)
    fraction_confident_irrelevant: Mapped[float] = mapped_column(Float, nullable=False)

    # Learning dynamics
    mean_abs_delta_mean: Mapped[float] = mapped_column(Float, nullable=False)
    mean_delta_precision: Mapped[float] = mapped_column(Float, nullable=False)
    fraction_weights_updated: Mapped[float] = mapped_column(Float, nullable=False)

    # Interpretability (legacy interactions removed)
    top_certain_weights: Mapped[Optional[List[Any]]] = mapped_column(
        JSON, nullable=True
    )  # Top 10 most certain (high precision, high contribution)
    top_uncertain_weights: Mapped[Optional[List[Any]]] = mapped_column(
        JSON, nullable=True
    )  # Top 10 most uncertain (low precision, high contribution)
    top_importance_weights: Mapped[Optional[List[Any]]] = mapped_column(
        JSON, nullable=True
    )  # Top 10 by actual contribution to predictions
