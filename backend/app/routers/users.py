from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import UserStats
from app.models.db_models import TypingSession, User
from sqlalchemy import func
import uuid

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
    ).filter(TypingSession.user_id == user_id).first()

    if not stats or stats.total_sessions == 0:
        raise HTTPException(status_code=404, detail="No stats found for this user.")

    return UserStats(
        total_sessions=stats.total_sessions,
        avg_wpm=stats.avg_wpm or 0.0,
        avg_accuracy=stats.avg_accuracy or 0.0
    )
