from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import SessionCreateRequest, SessionResponse
from app.models.db_models import TypingSession
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=SessionResponse)
def create_session(
    request: SessionCreateRequest, 
    db: Session = Depends(get_db)
):
    """
    Saves a typing session to the database.
    """
    # A real implementation would calculate flow_score here
    flow_score = request.wpm * request.accuracy * (request.difficultyLevel / 5.0)

    db_session = TypingSession(
        user_id=request.user_id,
        duration_seconds=request.durationSeconds,
        words_typed=request.wordsTyped,
        characters_typed=len("".join(k.key for k in request.keystrokeData if not k.isBackspace)), # Approximation
        errors=request.errors,
        backspaces=sum(1 for k in request.keystrokeData if k.isBackspace),
        final_wpm=request.wpm,
        avg_wpm=request.wpm, # Placeholder
        peak_wpm=request.wpm, # Placeholder
        accuracy=request.accuracy,
        starting_difficulty=request.difficultyLevel, # Placeholder
        ending_difficulty=request.difficultyLevel,
        avg_difficulty=request.difficultyLevel,
        keystroke_events=[k.dict() for k in request.keystrokeData],
        flow_score=flow_score
    )
    
    try:
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        logger.info(f"Saved session {db_session.id} with flow_score {flow_score}")
        return db_session
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save session: {e}")
        raise
