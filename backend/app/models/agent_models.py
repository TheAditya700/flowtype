"""Database models for LinTS agent observability."""

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Index
from sqlalchemy.sql import func
from app.database import Base
import uuid


class AgentWeightSnapshot(Base):
    """Periodic snapshots of the agent's weight matrix statistics.

    Captures the state of W_mean and W_precision at regular intervals
    to track learning dynamics and convergence.
    """

    __tablename__ = "agent_weight_snapshots"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=func.now(), index=True)

    # Tracking metadata
    update_count = Column(Integer, nullable=False)
    session_count = Column(Integer, nullable=False)

    # Weight matrix statistics (Global)
    mean_abs = Column(Float)  # Average magnitude of weights
    mean_std = Column(Float)

    # Precision statistics (Global confidence)
    precision_mean = Column(Float)
    precision_min = Column(Float)
    precision_max = Column(Float)

    # Exploration metrics
    avg_variance = Column(Float)  # 1/precision
    entropy_proxy = Column(Float)  # Variance of recent query vectors

    # Sampled weights for "Mean vs Uncertainty" scatter plot
    # Stored as list of [mean_val, precision_val] pairs (e.g. 100 random samples)
    weight_samples = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_snapshot_created", "created_at"),
        Index("idx_snapshot_update_count", "update_count"),
    )


class AgentRewardHistory(Base):
    """Individual reward signals from agent updates."""

    __tablename__ = "agent_reward_history"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=func.now(), index=True)

    # Context
    session_id = Column(String(36), index=True, nullable=True)
    user_id = Column(String(36), index=True, nullable=True)
    snippet_id = Column(String(36), index=True, nullable=True)

    # Total reward
    reward = Column(Float, nullable=False)

    # Reward decomposition (The "Why")
    reward_accuracy = Column(Float)  # w1 * delta_A
    reward_consistency = Column(Float)  # w2 * delta_A * delta_C
    reward_speed = Column(Float)  # w3 * delta_A * delta_C * delta_S

    # Deltas (The "What happened")
    delta_accuracy = Column(Float)
    delta_consistency = Column(Float)
    delta_speed = Column(Float)

    # Baselines (The "Expectation")
    baseline_accuracy = Column(Float)
    baseline_eff_wpm = Column(Float)

    # Agent state at time of update
    agent_update_count = Column(Integer)

    __table_args__ = (
        Index("idx_reward_created", "created_at"),
        Index("idx_reward_session", "session_id"),
    )
