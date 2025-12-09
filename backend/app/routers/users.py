from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import UserStats, UserProfile
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
        func.avg(TypingSession.final_wpm).label("avg_wpm"),
        func.avg(TypingSession.accuracy).label("avg_accuracy"),
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
        val = db.query(func.max(TypingSession.final_wpm)).filter(
            and_(
                TypingSession.user_id == user_id,
                TypingSession.duration_seconds >= target_time - 1,
                TypingSession.duration_seconds <= target_time + 1
            )
        ).scalar()
        return val or 0.0

    return UserStats(
        total_sessions=basic_stats.total_sessions if basic_stats else 0,
        avg_wpm=basic_stats.avg_wpm or 0.0,
        avg_accuracy=basic_stats.avg_accuracy or 0.0,
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
