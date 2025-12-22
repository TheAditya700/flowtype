from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Dict, Any, Optional
from app.database import SessionLocal
from app.models.agent_models import AgentWeightSnapshot, AgentRewardHistory
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Schemas ---

class WeightSnapshotResponse(BaseModel):
    id: str
    created_at: datetime
    update_count: int
    mean_abs: float
    mean_std: float
    precision_mean: float
    avg_variance: float
    entropy_proxy: Optional[float] = 0.0
    weight_samples: Optional[List[List[float]]] = None # [[mean, precision], ...]

    class Config:
        from_attributes = True

class RewardHistoryResponse(BaseModel):
    id: str
    created_at: datetime
    reward: float
    reward_accuracy: Optional[float]
    reward_consistency: Optional[float]
    reward_speed: Optional[float]
    delta_accuracy: Optional[float]
    delta_consistency: Optional[float]
    delta_speed: Optional[float]
    baseline_accuracy: Optional[float]
    baseline_eff_wpm: Optional[float]
    agent_update_count: Optional[int]

    class Config:
        from_attributes = True

# --- Endpoints ---

@router.get("/agent/snapshots", response_model=List[WeightSnapshotResponse])
def get_agent_snapshots(
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """Get recent agent weight snapshots for monitoring learning progress."""
    snapshots = db.query(AgentWeightSnapshot)\
        .order_by(desc(AgentWeightSnapshot.update_count))\
        .limit(limit)\
        .all()
    return snapshots

@router.get("/agent/rewards", response_model=List[RewardHistoryResponse])
def get_agent_rewards(
    limit: int = 200, 
    db: Session = Depends(get_db)
):
    """Get recent reward history for reward decomposition analysis."""
    rewards = db.query(AgentRewardHistory)\
        .order_by(desc(AgentRewardHistory.created_at))\
        .limit(limit)\
        .all()
    return rewards

@router.get("/agent/stats/summary")
def get_agent_summary(db: Session = Depends(get_db)):
    """Get high-level summary stats for the dashboard header."""
    
    # Latest snapshot
    latest_snapshot = db.query(AgentWeightSnapshot)\
        .order_by(desc(AgentWeightSnapshot.update_count))\
        .first()
        
    # Total updates
    total_updates = 0
    if latest_snapshot:
        total_updates = latest_snapshot.update_count
        
    # Average reward (last 100)
    avg_reward = db.query(func.avg(AgentRewardHistory.reward))\
        .order_by(desc(AgentRewardHistory.created_at))\
        .limit(100)\
        .scalar() or 0.0
        
    return {
        "total_updates": total_updates,
        "current_precision_mean": latest_snapshot.precision_mean if latest_snapshot else 0.0,
        "current_entropy": latest_snapshot.entropy_proxy if latest_snapshot else 0.0,
        "recent_avg_reward": float(avg_reward)
    }
