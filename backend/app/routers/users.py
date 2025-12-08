from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import UserStats, UserProfile
from app.models.db_models import TypingSession, User
from sqlalchemy import func, and_
import uuid

from app.routers.auth import get_current_active_user # Import the dependency

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _calculate_stats(user_id: str, db: Session) -> UserStats:
    # Basic aggregates
    basic_stats = db.query(
        func.count(TypingSession.id).label("total_sessions"),
        func.avg(TypingSession.final_wpm).label("avg_wpm"),
        func.avg(TypingSession.accuracy).label("avg_accuracy"),
        func.sum(TypingSession.duration_seconds).label("total_time")
    ).filter(TypingSession.user_id == user_id).first()

    # Best WPMs for specific durations (approximate matching)
    def get_best_wpm(target_time: int):
        return db.query(func.max(TypingSession.final_wpm)).filter(
            and_(
                TypingSession.user_id == user_id,
                TypingSession.duration_seconds >= target_time - 1,
                TypingSession.duration_seconds <= target_time + 1
            )
        ).scalar()

    return UserStats(
        total_sessions=basic_stats.total_sessions if basic_stats else 0,
        avg_wpm=basic_stats.avg_wpm or 0.0,
        avg_accuracy=basic_stats.avg_accuracy or 0.0,
        total_time_typing=basic_stats.total_time or 0.0,
        best_wpm_15=get_best_wpm(15) or 0.0,
        best_wpm_30=get_best_wpm(30) or 0.0,
        best_wpm_60=get_best_wpm(60) or 0.0,
        best_wpm_120=get_best_wpm(120) or 0.0
    )

@router.get("/{user_id}/stats", response_model=UserStats)
def get_user_stats(user_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Retrieves aggregate typing statistics for a given user.
    """
    return _calculate_stats(str(user_id), db)


@router.get("/me/profile", response_model=UserProfile) # Change path and remove user_id param
async def get_user_profile(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """
    Retrieves the full user profile including skill features for the authenticated user.
    """
    # current_user is already loaded by the dependency
    user_id = str(current_user.id) # Get ID from the authenticated user
    
    stats = _calculate_stats(user_id, db)

    return UserProfile(
        user_id=user_id,
        username=current_user.username,
        features=current_user.features or {},
        stats=stats
    )
