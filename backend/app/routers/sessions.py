# ----------------------------
# sessions.py  (rewritten)
# ----------------------------
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import SessionCreateRequest, SessionResponse
from app.models.db_models import TypingSession, SnippetUsage, KeystrokeEventDB, User
from app.ml.user_features import UserFeatureExtractor
from sqlalchemy.sql import func
import logging, uuid

router = APIRouter()
logger = logging.getLogger(__name__)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/", response_model=SessionResponse)
def create_session(request: SessionCreateRequest, db: Session = Depends(get_db)):
    """
    Handles the end of a snippet typing session:
      - Saves the final session row
      - Saves snippet usage rows
      - Saves keystrokes
      - Updates User long-term stats (via UserFeatureExtractor)
      - Computes reward
    """
    try:
        # -----------------------------------------------------
        # Compute basic reward (RL training)
        # -----------------------------------------------------
        reward = (request.wpm * request.accuracy) - (0.5 * request.errors)

        # -----------------------------------------------------
        # Insert TypingSession row
        # -----------------------------------------------------
        db_session = TypingSession(
            user_id=request.user_id,
            duration_seconds=request.durationSeconds,
            words_typed=request.wordsTyped,
            errors=request.errors,
            backspaces=sum(1 for k in request.keystrokeData if k.isBackspace),
            final_wpm=request.wpm,
            accuracy=request.accuracy,
            avg_difficulty=request.difficultyLevel,
            starting_difficulty=request.difficultyLevel,
            ending_difficulty=request.difficultyLevel,
            flow_score=request.flowScore,
            reward=reward,
        )
        db.add(db_session)
        db.flush()

        # -----------------------------------------------------
        # Save Keystrokes (for GRU training)
        # -----------------------------------------------------
        kevent_objs = [
            KeystrokeEventDB(
                session_id=db_session.id,
                timestamp=e.timestamp,
                key=e.key,
                is_backspace=e.isBackspace,
                is_correct=e.isCorrect,
            )
            for e in request.keystrokeData
        ]
        db.add_all(kevent_objs)

        # -----------------------------------------------------
        # Save SnippetUsage rows
        # -----------------------------------------------------
        usage_objs = []
        for idx, s in enumerate(request.snippets):
            usage_objs.append(
                SnippetUsage(
                    id=str(uuid.uuid4()),
                    session_id=db_session.id,
                    snippet_id=s.snippet_id,
                    user_wpm=s.wpm,
                    user_accuracy=s.accuracy,
                    snippet_position=idx,
                    difficulty_snapshot=s.difficulty,
                )
            )
        db.add_all(usage_objs)

        # -----------------------------------------------------
        # Update User Long-Term Stats (UserFeatureExtractor)
        # -----------------------------------------------------
        if request.user_id:
            user = db.query(User).filter(User.id == request.user_id).first()
            
            # Create user if not exists (lazy creation for no-auth flow)
            if not user:
                user = User(id=request.user_id)
                db.add(user)
                # Flush to ensure ID is available if needed, though we set it manually
                db.flush() 

            # Load extractor from JSON (or init new)
            extractor = UserFeatureExtractor.from_dict(user.features or {})
            
            # Prepare session dict for extractor
            # UserFeatureExtractor expects dict-based events with keys matching KeystrokeEvent schema
            events_dicts = [
                {
                    "key": k.key,
                    "isBackspace": k.isBackspace,
                    "isCorrect": k.isCorrect,
                    "timestamp": k.timestamp,
                    "keyup_timestamp": k.keyup_timestamp
                } 
                for k in request.keystrokeData
            ]
            
            session_data = {
                'keystroke_events': events_dicts,
                'wpm': request.wpm,
                'accuracy': request.accuracy,
                'snippet_difficulty': request.difficultyLevel,
                'completed': True,  # Submitting a session implies completion/submission
                'quit_progress': 1.0
            }
            
            # Update and Save
            extractor.update_from_session(session_data)
            user.features = extractor.to_dict()
            
            # Update last active
            user.last_active = func.now()
            db.add(user)

        db.commit()
        db.refresh(db_session)

        response = SessionResponse(
            session_id=str(db_session.id),
            reward=reward
        )

        return response

    except Exception as e:
        db.rollback()
        logger.exception("Failed to save session")
        raise HTTPException(status_code=500, detail="Failed to save session")
