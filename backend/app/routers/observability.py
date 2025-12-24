from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
import numpy as np

from app.database import SessionLocal
from app.models.db_models import TypingSession, ModelSnapshots, User


router = APIRouter()


class ScaleEnum(str, Enum):
    single = "single"
    x10 = "x10"
    x100 = "x100"
    x1000 = "x1000"


POINTS_PER_VIEW = 10


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _group_size(scale: ScaleEnum) -> int:
    mapping = {
        ScaleEnum.single: 1,
        ScaleEnum.x10: 10,
        ScaleEnum.x100: 100,
        ScaleEnum.x1000: 1000,
    }
    return mapping.get(scale, 1)


def _effective_limit(scale: ScaleEnum, limit: Optional[int]) -> int:
    return limit if limit is not None else POINTS_PER_VIEW * _group_size(scale)


def _aggregate_mean(
    points: List[Dict[str, Any]], group_size: int, fields: List[str]
) -> List[Dict[str, Any]]:
    """Chunk sequential points and average numeric fields per chunk."""
    if not points:
        return []

    aggregated: List[Dict[str, Any]] = []
    for idx in range(0, len(points), group_size):
        chunk = points[idx : idx + group_size]
        if not chunk:
            continue

        agg: Dict[str, Any] = {"t": chunk[-1].get("t")}
        for field in fields:
            values = [float(item.get(field, 0.0)) for item in chunk]
            agg[field] = float(np.mean(values)) if values else 0.0

        aggregated.append(agg)

    return aggregated


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
    one_day_ago = datetime.now(timezone.utc) - timedelta(days=1)
    active_users = (
        db.query(TypingSession.user_id)
        .filter(TypingSession.created_at >= one_day_ago)
        .filter(TypingSession.user_id.isnot(None))
        .distinct()
        .count()
    )
    sessions_last_24h = (
        db.query(TypingSession).filter(TypingSession.created_at >= one_day_ago).count()
    )

    # Latest snapshot info (keep model version; drop last_snapshot_time)
    latest_snapshot = (
        db.query(ModelSnapshots).order_by(ModelSnapshots.created_at.desc()).first()
    )

    model_version = latest_snapshot.model_version if latest_snapshot else "v0.0.0"

    return {
        "total_sessions": total_sessions,
        "active_users": active_users,
        "sessions_last_24h": sessions_last_24h,
        "model_version": model_version,
        # last_snapshot_time removed
    }


# ==============================================================================
# 2. LEARNING HEALTH (Precision & Variance over time)
# ==============================================================================
@router.get("/observability/learning_health")
def get_learning_health(
    db: Session = Depends(get_db),
    scale: ScaleEnum = Query(ScaleEnum.single),
    limit: Optional[int] = Query(None, ge=1, le=20000),
):
    """
    Returns session-scaled series of model learning health, chunked by scale:
    - mean_precision (primary indicator)
    - mean_variance (secondary axis)
    """
    effective_limit = _effective_limit(scale, limit)

    snapshots: List[ModelSnapshots] = (
        db.query(ModelSnapshots)
        .order_by(ModelSnapshots.created_at.desc())
        .limit(effective_limit)
        .all()
    )

    snapshots.reverse()  # Chronological

    raw_points = []
    for snap in snapshots:
        if snap.created_at:
            raw_points.append(
                {
                    "t": snap.created_at.isoformat(),
                    "mean_precision": float(snap.mean_precision or 0.0),
                    "mean_variance": float(snap.mean_variance or 0.0),
                }
            )

    grouped = _aggregate_mean(
        raw_points, _group_size(scale), ["mean_precision", "mean_variance"]
    )
    points = grouped[-POINTS_PER_VIEW:]

    return {"scale": scale.value, "points": points}


# ==============================================================================
# 3. AGENT EFFECTIVENESS (Reward over time)
# ==============================================================================
@router.get("/observability/agent_effectiveness")
def get_agent_effectiveness(
    db: Session = Depends(get_db),
    scale: ScaleEnum = Query(ScaleEnum.single),
    limit: Optional[int] = Query(None, ge=1, le=20000),
):
    """
    Returns session-scaled reward signal:
    - mean_reward (primary metric)
    - reward_variance / reward_std (consistency)
    Grouped by the requested session scale.
    """
    effective_limit = _effective_limit(scale, limit)

    sessions: List[TypingSession] = (
        db.query(TypingSession)
        .filter(TypingSession.reward.isnot(None))
        .order_by(TypingSession.created_at.desc())
        .limit(effective_limit)
        .all()
    )

    sessions.reverse()

    raw_points = []
    for s in sessions:
        if not s.created_at:
            continue

        reward = float(s.reward or 0.0)
        if not np.isfinite(reward):
            continue

        raw_points.append(
            {
                "t": s.created_at.isoformat(),
                "mean_reward": reward,
                "reward_variance": 0.0,  # computed per chunk
                "reward_std": 0.0,  # computed per chunk
                "count": 1,
            }
        )

    group_size = _group_size(scale)
    aggregated: List[Dict[str, Any]] = []
    for idx in range(0, len(raw_points), group_size):
        chunk = raw_points[idx : idx + group_size]
        if not chunk:
            continue

        rewards = [float(item["mean_reward"]) for item in chunk]

        # Compute std/var; if the chunk is too small (e.g., single-scale),
        # use a rolling window of up to the last 10 raw rewards to avoid zeroed std.
        if len(rewards) >= 2:
            win_std = float(np.std(rewards))
            win_var = float(np.var(rewards))
        else:
            end = min(idx + group_size, len(raw_points))
            w_start = max(0, end - 10)
            window_rewards = [
                float(item["mean_reward"]) for item in raw_points[w_start:end]
            ]
            if len(window_rewards) >= 2:
                win_std = float(np.std(window_rewards))
                win_var = float(np.var(window_rewards))
            else:
                win_std = 0.0
                win_var = 0.0

        aggregated.append(
            {
                "t": chunk[-1]["t"],
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "reward_variance": win_var,
                "reward_std": win_std,
                "count": sum(int(item.get("count", 1)) for item in chunk),
            }
        )

    points = aggregated[-POINTS_PER_VIEW:]

    return {"scale": scale.value, "points": points}


# ==============================================================================
# 4. PERFORMANCE DELTAS (What is improving?)
# ==============================================================================
@router.get("/observability/performance_deltas")
def get_performance_deltas(
    db: Session = Depends(get_db),
    scale: ScaleEnum = Query(ScaleEnum.single),
    limit: Optional[int] = Query(None, ge=1, le=20000),
):
    """
    Returns session-scaled performance deltas vs EMA baselines:
    - delta_accuracy (actual - EMA baseline)
    - delta_smoothness (actual - EMA baseline)
    - delta_effective_wpm (actual - EMA baseline)
    """
    effective_limit = _effective_limit(scale, limit)

    sessions: List[TypingSession] = (
        db.query(TypingSession)
        .order_by(TypingSession.created_at.desc())
        .limit(effective_limit)
        .all()
    )

    sessions.reverse()  # Chronological

    # Fetch user EMA baselines (cached lookup)
    user_emas: Dict[str, Optional[List[float]]] = {}

    raw_points = []
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
        base_smoothness = 0.5 * (1.0 / (1.0 + base_iki_cv)) + 0.5 * (
            1.0 - base_spike_rate
        )

        # Current effective WPM
        actual_eff_wpm = actual_wpm * actual_acc

        delta_acc = actual_acc - base_acc
        delta_smoothness = actual_consistency - base_smoothness
        delta_eff_wpm = actual_eff_wpm - base_eff_wpm

        raw_points.append(
            {
                "t": s.created_at.isoformat(),
                "delta_accuracy": float(delta_acc),
                "delta_smoothness": float(delta_smoothness),
                "delta_effective_wpm": float(delta_eff_wpm),
                "actual_accuracy": float(actual_acc),
                "actual_consistency": float(actual_consistency),
                "actual_effective_wpm": float(actual_eff_wpm),
            }
        )

    grouped = _aggregate_mean(
        raw_points,
        _group_size(scale),
        [
            "delta_accuracy",
            "delta_smoothness",
            "delta_effective_wpm",
            "actual_accuracy",
            "actual_consistency",
            "actual_effective_wpm",
        ],
    )
    points = grouped[-POINTS_PER_VIEW:]

    return {"scale": scale.value, "points": points}


# New: return all three lists in one response
@router.get("/observability/user_skills_all")
def get_user_skills_all(
    db: Session = Depends(get_db),
    top_k: int = Query(10, ge=1, le=50),
):
    latest_snapshot = (
        db.query(ModelSnapshots).order_by(ModelSnapshots.created_at.desc()).first()
    )

    if not latest_snapshot:
        return {"impact": [], "certain": [], "uncertain": []}

    impact = (latest_snapshot.top_importance_weights or [])[:top_k]
    certain = (latest_snapshot.top_certain_weights or [])[:top_k]
    uncertain = (latest_snapshot.top_uncertain_weights or [])[:top_k]

    return {
        "impact": impact,
        "certain": certain,
        "uncertain": uncertain,
    }


# ==============================================================================
# 6. LEARNING ACTIVITY (Is learning done or still active?)
# ==============================================================================
@router.get("/observability/learning_activity")
def get_learning_activity(
    db: Session = Depends(get_db),
    scale: ScaleEnum = Query(ScaleEnum.single),
    limit: Optional[int] = Query(None, ge=1, le=20000),
):
    """
    Returns session-scaled learning dynamics:
    - mean_abs_delta_mean (how much weights are changing)
    - fraction_weights_updated (what % of weights changed)
    """
    effective_limit = _effective_limit(scale, limit)

    snapshots: List[ModelSnapshots] = (
        db.query(ModelSnapshots)
        .order_by(ModelSnapshots.created_at.desc())
        .limit(effective_limit)
        .all()
    )

    snapshots.reverse()

    raw_points = []
    for snap in snapshots:
        if not snap.created_at:
            continue

        if snap.mean_abs_delta_mean == 0.0 and snap.fraction_weights_updated == 0.0:
            continue

        raw_points.append(
            {
                "t": snap.created_at.isoformat(),
                "mean_abs_delta_mean": float(snap.mean_abs_delta_mean or 0.0),
                "fraction_weights_updated": float(snap.fraction_weights_updated or 0.0),
            }
        )

    grouped = _aggregate_mean(
        raw_points,
        _group_size(scale),
        ["mean_abs_delta_mean", "fraction_weights_updated"],
    )
    points = grouped[-POINTS_PER_VIEW:]

    return {"scale": scale.value, "points": points}
