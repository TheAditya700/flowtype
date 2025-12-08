from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import UserStats, UserProfile
from app.models.db_models import TypingSession, User
from sqlalchemy import func
import uuid

from app.routers.auth import get_current_active_user # Import the dependency

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/{user_id}/stats", response_model=UserStats)
def get_user_stats(user_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Retrieves aggregate typing statistics for a given user.
    """
    stats = db.query(
        func.count(TypingSession.id).label("total_sessions"),
        func.avg(TypingSession.final_wpm).label("avg_wpm"),
        func.avg(TypingSession.accuracy).label("avg_accuracy")
    ).filter(TypingSession.user_id == str(user_id)).first()

    if not stats or stats.total_sessions == 0:
        # Return empty stats instead of 404 for better UX
        return UserStats(
            total_sessions=0,
            avg_wpm=0.0,
            avg_accuracy=0.0
        )

    return UserStats(
        total_sessions=stats.total_sessions,
        avg_wpm=stats.avg_wpm or 0.0,
        avg_accuracy=stats.avg_accuracy or 0.0
    )


@router.get("/me/profile", response_model=UserProfile) # Change path and remove user_id param
async def get_user_profile(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """
    Retrieves the full user profile including skill features for the authenticated user.
    """
    # current_user is already loaded by the dependency
    user_id = str(current_user.id) # Get ID from the authenticated user
    
    # Calculate stats (reuse logic or call internal function)
    stats_query = db.query(
        func.count(TypingSession.id).label("total_sessions"),
        func.avg(TypingSession.final_wpm).label("avg_wpm"),
        func.avg(TypingSession.accuracy).label("avg_accuracy")
    ).filter(TypingSession.user_id == user_id).first()

    stats = UserStats(
        total_sessions=stats_query.total_sessions if stats_query else 0,
        avg_wpm=stats_query.avg_wpm or 0.0,
        avg_accuracy=stats_query.avg_accuracy or 0.0
    )

    return UserProfile(
        user_id=user_id,
        username=current_user.username,
        features=current_user.features or {},
        stats=stats
    )
