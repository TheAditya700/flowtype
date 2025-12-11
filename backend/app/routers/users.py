from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import UserStats, UserProfile, UserStatsDetail, SessionTimeseriesPoint, ActivityDay, CharStat, LeaderboardEntry
from app.models.db_models import TypingSession, User
from sqlalchemy import func, and_
import uuid
from typing import Optional

from app.routers.auth import get_current_active_user # Import the dependency

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _calculate_stats(user_id: str, db: Session, user_obj: Optional[User] = None) -> UserStats:
    # 1. Fetch User if not provided to access cache
    if not user_obj:
        try:
            # Handle potential UUID vs String mismatch
            user_obj = db.query(User).filter(User.id == user_id).first()
        except:
            pass

    # 2. Basic aggregates (Always query DB for these as they change every session)
    basic_stats = db.query(
        func.count(TypingSession.id).label("total_sessions"),
        func.avg(TypingSession.actual_wpm).label("avg_wpm"),
        func.avg(TypingSession.actual_accuracy).label("avg_accuracy"),
        func.sum(TypingSession.duration_seconds).label("total_time")
    ).filter(TypingSession.user_id == user_id).first()

    # 3. Best WPMs (Use Cache if available)
    best_wpms_cache = user_obj.best_wpms if user_obj and user_obj.best_wpms else {}

    def get_best_wpm(target_time: int):
        # Check cache first
        cached = float(best_wpms_cache.get(str(target_time), 0.0))
        if cached > 0:
            return cached
            
        # Fallback to expensive query
        val = db.query(func.max(TypingSession.actual_wpm)).filter(
            and_(
                TypingSession.user_id == user_id,
                TypingSession.duration_seconds >= target_time - 1,
                TypingSession.duration_seconds <= target_time + 1
            )
        ).scalar()
        return val or 0.0

    avg_accuracy_pct = (basic_stats.avg_accuracy or 0.0) * 100.0

    return UserStats(
        total_sessions=basic_stats.total_sessions if basic_stats else 0,
        avg_wpm=basic_stats.avg_wpm or 0.0,
        avg_accuracy=avg_accuracy_pct,
        total_time_typing=basic_stats.total_time or 0.0,
        best_wpm_15=get_best_wpm(15),
        best_wpm_30=get_best_wpm(30),
        best_wpm_60=get_best_wpm(60),
        best_wpm_120=get_best_wpm(120)
    )

@router.get("/{user_id}/stats", response_model=UserStats)
def get_user_stats(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieves aggregate typing statistics for a given user.
    """
    return _calculate_stats(user_id, db)


@router.get("/{user_id}/stats/detail", response_model=UserStatsDetail)
def get_user_stats_detail(user_id: str, db: Session = Depends(get_db)):
    """
    Detailed stats including EMA series and activity heatmap.
    """
    summary = _calculate_stats(user_id, db)

    sessions = db.query(TypingSession).filter(TypingSession.user_id == user_id).order_by(TypingSession.created_at.asc()).all()

    timeseries = []
    alpha = 0.3
    ema_wpm = None
    ema_wpm_dev = None
    ema_acc = None
    for s in sessions:
        wpm = float(s.actual_wpm or 0.0)
        acc = float((s.actual_accuracy or 0.0) * 100.0)
        raw = float(s.raw_wpm or 0.0)
        ts = int(s.created_at.timestamp() * 1000) if s.created_at else 0
        if ema_wpm is None:
            ema_wpm = wpm
            ema_wpm_dev = 0.0
        else:
            diff = abs(wpm - ema_wpm)
            ema_wpm = alpha * wpm + (1 - alpha) * ema_wpm
            ema_wpm_dev = alpha * diff + (1 - alpha) * (ema_wpm_dev or 0.0)

        if ema_acc is None:
            ema_acc = acc
        else:
            ema_acc = alpha * acc + (1 - alpha) * ema_acc

        timeseries.append(SessionTimeseriesPoint(
            timestamp=ts,
            wpm=wpm,
            accuracy=acc,
            raw_wpm=raw,
            ema_wpm=ema_wpm,
            ema_dev=ema_wpm_dev,
            ema_accuracy=ema_acc
        ))

    # Activity heatmap per day
    activity_map = {}
    for s in sessions:
        if not s.created_at:
            continue
        day = s.created_at.date().isoformat()
        activity_map[day] = activity_map.get(day, 0) + 1

    activity = [ActivityDay(date=k, count=v) for k, v in sorted(activity_map.items())]

    # Streak calculations
    days_sorted = sorted(activity_map.keys())
    current_streak = 0
    longest_streak = 0
    if days_sorted:
        from datetime import datetime, timedelta
        day_set = set(days_sorted)
        today = datetime.utcnow().date()

        # Current streak: count backwards from today
        cursor = today
        while cursor.isoformat() in day_set:
            current_streak += 1
            cursor -= timedelta(days=1)

        # Longest streak
        longest_streak = current_streak
        streak = 0
        prev = None
        for day_str in days_sorted:
            day = datetime.fromisoformat(day_str).date()
            if prev and (day - prev).days == 1:
                streak += 1
            else:
                streak = 1
            longest_streak = max(longest_streak, streak)
            prev = day

    # Character heatmap across all sessions (accuracy + relative frequency as speed proxy)
    char_totals = {}
    char_correct = {}
    for s in sessions:
        events = s.keystroke_events or []
        if not isinstance(events, list):
            continue
        for ev in events:
            try:
                key = (ev.get("key") or "").lower()
                if not key or len(key) != 1:
                    continue
                char_totals[key] = char_totals.get(key, 0) + 1
                if ev.get("isCorrect"):
                    char_correct[key] = char_correct.get(key, 0) + 1
            except Exception:
                continue

    max_count = max(char_totals.values()) if char_totals else 1
    char_heatmap = {
        k: CharStat(
            accuracy=(char_correct.get(k, 0) / v) if v else 0.0,
            speed=min(1.0, v / max_count)
        )
        for k, v in char_totals.items()
    }

    return UserStatsDetail(
        summary=summary,
        timeseries=timeseries,
        activity=activity,
        current_streak=current_streak,
        longest_streak=longest_streak,
        char_heatmap=char_heatmap,
    )

@router.get("/leaderboard", response_model=list[LeaderboardEntry])
def get_leaderboard(mode: str = "60", exclude_anon: bool = False, db: Session = Depends(get_db)):
    """
    All-time leaderboard for a timed mode (15/30/60/120). Uses cached best_wpms when available,
    with a DB fallback per user if missing. Returns top 100.
    If exclude_anon is True, filters out anonymous users (users without username).
    """
    valid_modes = {"15", "30", "60", "120"}
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail="Invalid mode; must be one of 15,30,60,120")

    users = db.query(User).all()
    entries: list[LeaderboardEntry] = []
    target = int(mode)

    for u in users:
        # Skip anonymous users if filter is enabled
        if exclude_anon and not u.username:
            continue
            
        best_dict = u.best_wpms or {}
        best = float(best_dict.get(mode, 0.0))

        if best <= 0.0:
            best = db.query(func.max(TypingSession.actual_wpm)).filter(
                and_(
                    TypingSession.user_id == u.id,
                    TypingSession.duration_seconds >= target - 1,
                    TypingSession.duration_seconds <= target + 1
                )
            ).scalar() or 0.0

        entries.append(LeaderboardEntry(
            user_id=u.id,
            username=u.username,
            best_wpm=best,
            mode=mode
        ))

    entries.sort(key=lambda e: e.best_wpm, reverse=True)
    return entries[:100]


@router.get("/me/profile", response_model=UserProfile)
async def get_user_profile(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """
    Retrieves the full user profile including skill features for the authenticated user.
    """
    user_id = str(current_user.id)
    
    # Pass current_user object to leverage cached best_wpms
    stats = _calculate_stats(user_id, db, user_obj=current_user)

    return UserProfile(
        user_id=user_id,
        username=current_user.username,
        features=current_user.features or {},
        stats=stats
    )
