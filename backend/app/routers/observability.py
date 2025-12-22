from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from enum import Enum
import numpy as np

from app.database import SessionLocal
from app.models.db_models import TypingSession, ModelSnapshots, User


router = APIRouter()


class TimeframeEnum(str, Enum):
    minute = "minute"
    hour = "hour"
    day = "day"
    week = "week"
    month = "month"
    year = "year"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_trunc_func(timeframe: TimeframeEnum) -> str:
    """Map timeframe to PostgreSQL date_trunc function."""
    mapping = {
        TimeframeEnum.minute: "minute",
        TimeframeEnum.hour: "minute",
        TimeframeEnum.day: "hour",
        TimeframeEnum.week: "day",
        TimeframeEnum.month: "day",
        TimeframeEnum.year: "month",
    }
    return mapping.get(timeframe, "day")


def _get_lookback_timedelta(timeframe: TimeframeEnum) -> timedelta:
    """Get sensible lookback period for timeframe."""
    mapping = {
        TimeframeEnum.minute: timedelta(minutes=60),
        TimeframeEnum.hour: timedelta(hours=1),
        TimeframeEnum.day: timedelta(days=1),
        TimeframeEnum.week: timedelta(days=7),
        TimeframeEnum.month: timedelta(days=30),
        TimeframeEnum.year: timedelta(days=365),
    }
    return mapping.get(timeframe, timedelta(days=1))


# ==============================================================================
# 1. HEADER (KPIs)
# ==============================================================================
@router.get("/observability/header")
def get_observability_header(
    db: Session = Depends(get_db),
):
    """
    Returns high-level KPIs for the observability dashboard header.
    """
    # Total sessions
    total_sessions = db.query(TypingSession).count()
    
    # Active users (unique users with sessions in last 24h)
    one_day_ago = datetime.utcnow() - timedelta(days=1)
    active_users = (
        db.query(TypingSession.user_id)
        .filter(TypingSession.created_at >= one_day_ago)
        .filter(TypingSession.user_id.isnot(None))
        .distinct()
        .count()
    )
    
    # Latest snapshot info
    latest_snapshot = (
        db.query(ModelSnapshots)
        .order_by(ModelSnapshots.created_at.desc())
        .first()
    )
    
    model_version = latest_snapshot.model_version if latest_snapshot else "v0.0.0"
    last_snapshot_time = latest_snapshot.created_at.isoformat() if latest_snapshot else None
    
    return {
        "total_sessions": total_sessions,
        "active_users": active_users,
        "model_version": model_version,
        "last_snapshot_time": last_snapshot_time,
    }


# ==============================================================================
# 2. LEARNING HEALTH (Precision & Variance over time)
# ==============================================================================
@router.get("/observability/learning_health")
def get_learning_health(
    db: Session = Depends(get_db),
    timeframe: TimeframeEnum = Query(TimeframeEnum.day),
    limit: int = Query(100, ge=1, le=500),
):
    """
    Returns time-series of model learning health:
    - mean_precision (primary indicator)
    - mean_variance (secondary axis)
    
    Shows Bayesian learning progression (exploration → convergence).
    """
    lookback = _get_lookback_timedelta(timeframe)
    since = datetime.utcnow() - lookback

    snapshots: List[ModelSnapshots] = (
        db.query(ModelSnapshots)
        .filter(ModelSnapshots.created_at >= since)
        .order_by(ModelSnapshots.created_at.desc())
        .limit(limit)
        .all()
    )

    # Reverse to get chronological order for plotting
    snapshots.reverse()

    points = []
    for snap in snapshots:
        if snap.created_at:
            points.append({
                "t": snap.created_at.isoformat(),
                "mean_precision": float(snap.mean_precision or 0.0),
                "mean_variance": float(snap.mean_variance or 0.0),
            })

    return {"timeframe": timeframe.value, "points": points}


# ==============================================================================
# 3. AGENT EFFECTIVENESS (Reward over time)
# ==============================================================================
@router.get("/observability/agent_effectiveness")
def get_agent_effectiveness(
    db: Session = Depends(get_db),
    timeframe: TimeframeEnum = Query(TimeframeEnum.day),
    limit: int = Query(500, ge=1, le=1000),
):
    """
    Returns time-series of reward signal:
    - mean_reward (primary metric)
    - reward_variance (shows consistency)
    
    This is what the agent optimizes - clear success/failure signal.
    """
    # lookback logic
    lookback = _get_lookback_timedelta(timeframe)
    since = datetime.utcnow() - lookback

    sessions: List[TypingSession] = (
        db.query(TypingSession)
        .filter(TypingSession.created_at >= since)
        .filter(TypingSession.reward.isnot(None))
        .order_by(TypingSession.created_at.desc())
        .limit(limit)
        .all()
    )

    # Helper for python-side truncation
    def truncate_dt(dt: datetime, tf: TimeframeEnum) -> str:
        # Mapping timeframe -> truncation level
        # minute, hour -> minute (minute resolution)
        # day -> hour (hour resolution)
        # week -> day
        # month -> day
        if tf in (TimeframeEnum.minute, TimeframeEnum.hour):
            return dt.strftime("%Y-%m-%dT%H:%M:00")
        elif tf == TimeframeEnum.day:
            return dt.strftime("%Y-%m-%dT%H:00:00")
        elif tf in (TimeframeEnum.week, TimeframeEnum.month):
             return dt.strftime("%Y-%m-%dT00:00:00")
        else:
             # Fallback
             return dt.isoformat()

    # Aggregate by timeframe in Python
    agg: Dict[str, List[float]] = {}
    for s in sessions:
        if not s.created_at:
            continue
        
        # Truncate in Python
        key = truncate_dt(s.created_at, timeframe)
        
        if key not in agg:
            agg[key] = []
        
        reward = float(s.reward or 0.0)
        if np.isfinite(reward):
            agg[key].append(reward)

    # Compute mean and variance per bucket
    points = []
    for ts in sorted(agg.keys()):
        rewards = agg[ts]
        if rewards:
            points.append({
                "t": ts,
                "mean_reward": float(np.mean(rewards)),
                "reward_variance": float(np.var(rewards)),
                "reward_std": float(np.std(rewards)),
                "count": len(rewards),
            })

    return {"timeframe": timeframe.value, "points": points}


# ==============================================================================
# 4. PERFORMANCE DELTAS (What is improving?)
# ==============================================================================
@router.get("/observability/performance_deltas")
def get_performance_deltas(
    db: Session = Depends(get_db),
    timeframe: TimeframeEnum = Query(TimeframeEnum.day),
    limit: int = Query(500, ge=1, le=1000),
):
    """
    Returns time-series of performance deltas vs EMA baselines:
    - delta_accuracy (actual - EMA baseline)
    - delta_smoothness (actual - EMA baseline)
    - delta_effective_wpm (actual - EMA baseline)
    
    Shows what skills are improving over time.
    """
    lookback = _get_lookback_timedelta(timeframe)
    since = datetime.utcnow() - lookback

    sessions: List[TypingSession] = (
        db.query(TypingSession)
        .filter(TypingSession.created_at >= since)
        .order_by(TypingSession.created_at.desc())
        .limit(limit)
        .all()
    )
    
    sessions.reverse() # Chronological

    # Fetch user EMA baselines (cached lookup)
    user_emas: Dict[str, Optional[List[float]]] = {}

    points = []
    for s in sessions:
        if not s.created_at or not s.user_id:
            continue
        
        # Get user EMA baseline
        if s.user_id not in user_emas:
            user = db.query(User).filter(User.id == s.user_id).first()
            if user and user.features:
                ema_data = user.features.get("ema", {})
                user_emas[s.user_id] = ema_data.get("ema_mean", [])
            else:
                user_emas[s.user_id] = None
        
        ema_mean = user_emas[s.user_id]
        if not ema_mean or len(ema_mean) < 57:
            continue
        
        # Compute deltas
        actual_acc = float(s.actual_accuracy or 0.0)
        actual_wpm = float(s.actual_wpm or 0.0)
        actual_consistency = float(s.actual_consistency or 0.0) / 100.0  # normalize
        
        base_acc = float(ema_mean[0])  # IDX_ACCURACY
        base_wpm_raw = float(ema_mean[21])  # IDX_WPM_RAW
        base_eff_wpm = float(ema_mean[22])  # IDX_WPM_EFFECTIVE
        
        # Smoothness baseline: 0.5 * (1/(1+IKI_CV)) + 0.5 * (1-spike_rate)
        # We'll use IKI_CV at index 11 and spike_rate at index 28
        base_iki_cv = float(ema_mean[11]) if len(ema_mean) > 11 else 0.25
        base_spike_rate = float(ema_mean[28]) if len(ema_mean) > 28 else 0.20
        base_smoothness = 0.5 * (1.0 / (1.0 + base_iki_cv)) + 0.5 * (1.0 - base_spike_rate)
        
        # Current effective WPM
        actual_eff_wpm = actual_wpm * actual_acc
        
        delta_acc = actual_acc - base_acc
        delta_smoothness = actual_consistency - base_smoothness
        delta_eff_wpm = actual_eff_wpm - base_eff_wpm
        
        points.append({
            "t": s.created_at.isoformat(),
            "delta_accuracy": float(delta_acc),
            "delta_smoothness": float(delta_smoothness),
            "delta_effective_wpm": float(delta_eff_wpm),
            "actual_accuracy": float(actual_acc),
            "actual_consistency": float(actual_consistency),
            "actual_effective_wpm": float(actual_eff_wpm),
        })

    return {"timeframe": timeframe.value, "points": points}


# ==============================================================================
# 5. USER SKILLS IMPORTANCE (Top-K user features by weight)
# ==============================================================================
@router.get("/observability/user_skills")
def get_user_skills_importance(
    db: Session = Depends(get_db),
    top_k: int = Query(10, ge=1, le=20),
    mode: str = Query("importance", regex="^(importance|certain|uncertain)$"),
):
    """
    Returns top-K user features based on mode:
    - importance: Most impactful features (default, aggregated |W| across snippets)
    - certain: Most certain features (highest precision)
    - uncertain: Most uncertain features (lowest precision)
    
    Shows which user skills matter most for snippet selection.
    """
    latest_snapshot = (
        db.query(ModelSnapshots)
        .order_by(ModelSnapshots.created_at.desc())
        .first()
    )
    
    if not latest_snapshot:
        return {"skills": []}
    
    # Mode: Most Certain or Most Uncertain
    if mode == "certain":
        top_weights = latest_snapshot.top_certain_weights or []
        return {"skills": top_weights[:top_k]}
    
    if mode == "uncertain":
        top_weights = latest_snapshot.top_uncertain_weights or []
        return {"skills": top_weights[:top_k]}
    
    # Mode: Importance - use the new aggregated importance calculation
    if latest_snapshot.top_importance_weights:
        # New approach: use pre-computed importance
        return {"skills": latest_snapshot.top_importance_weights[:top_k]}
    
    # Fallback: original aggregation logic for backward compatibility
    # Get top positive and negative interactions
    top_pos = latest_snapshot.top_positive_interactions or []
    top_neg = latest_snapshot.top_negative_interactions or []
    
    # Aggregate importance by user feature index
    user_importance: Dict[int, Dict[str, Any]] = {}
    
    def process_interaction(interaction: Dict[str, Any], sign: str):
        user_idx = interaction.get("user_feature_idx")
        if user_idx is None:
            return
        
        if user_idx not in user_importance:
            user_importance[user_idx] = {
                "mean_weight": 0.0,
                "mean_precision": 0.0,
                "count": 0,
                "positive_count": 0,
                "negative_count": 0,
            }
        
        mean_val = abs(interaction.get("mean", 0.0))
        precision = interaction.get("precision", 0.0)
        
        user_importance[user_idx]["mean_weight"] += mean_val
        user_importance[user_idx]["mean_precision"] += precision
        user_importance[user_idx]["count"] += 1
        
        if sign == "positive":
            user_importance[user_idx]["positive_count"] += 1
        else:
            user_importance[user_idx]["negative_count"] += 1
    
    for interaction in top_pos:
        process_interaction(interaction, "positive")
    
    for interaction in top_neg:
        process_interaction(interaction, "negative")
    
    # Compute averages and sort by importance
    skills = []
    for user_idx, data in user_importance.items():
        count = data["count"]
        avg_weight = data["mean_weight"] / count if count > 0 else 0.0
        avg_precision = data["mean_precision"] / count if count > 0 else 0.0
        
        # Determine dominant sign
        if data["positive_count"] > data["negative_count"]:
            sign = "positive"
        elif data["negative_count"] > data["positive_count"]:
            sign = "negative"
        else:
            sign = "mixed"
        
        skills.append({
            "user_feature_idx": user_idx,
            "importance": float(avg_weight),
            "precision": float(avg_precision),
            "sign": sign,
            "interaction_count": count,
        })
    
    # Sort by importance and take top-K
    skills.sort(key=lambda x: x["importance"], reverse=True)
    
    return {"skills": skills[:top_k]}


# ==============================================================================
# 6. LEARNING ACTIVITY (Is learning done or still active?)
# ==============================================================================
@router.get("/observability/learning_activity")
def get_learning_activity(
    db: Session = Depends(get_db),
    timeframe: TimeframeEnum = Query(TimeframeEnum.day),
    limit: int = Query(100, ge=1, le=500),
):
    """
    Returns time-series of learning dynamics:
    - mean_abs_delta_mean (how much weights are changing)
    - fraction_weights_updated (what % of weights changed)
    
    Shows learning saturation or continued adaptation.
    """
    trunc = _get_trunc_func(timeframe)
    lookback = _get_lookback_timedelta(timeframe)
    since = datetime.utcnow() - lookback

    snapshots: List[ModelSnapshots] = (
        db.query(ModelSnapshots)
        .filter(ModelSnapshots.created_at >= since)
        .order_by(ModelSnapshots.created_at.desc())
        .limit(limit)
        .all()
    )
    
    snapshots.reverse()

    points = []
    for snap in snapshots:
        if snap.created_at:
            # Skip snapshots with no previous data (both values are zero)
            # This happens for the first snapshot where prev_snapshot is None
            if snap.mean_abs_delta_mean == 0.0 and snap.fraction_weights_updated == 0.0:
                continue
            
            points.append({
                "t": snap.created_at.isoformat(),
                "mean_abs_delta_mean": float(snap.mean_abs_delta_mean or 0.0),
                "fraction_weights_updated": float(snap.fraction_weights_updated or 0.0),
            })

    return {"timeframe": timeframe.value, "points": points}
