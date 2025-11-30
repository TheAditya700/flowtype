from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import SessionCreateRequest, SessionResponse
from app.models.db_models import TypingSession
from sqlalchemy import text
import uuid
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
        db.flush() # Get ID for session
        
        # Save snippet logs.
        # The DB schema may differ between environments (dev migrations not applied),
        # so insert snippet usage rows using raw SQL based on available columns.
        col_info = db.execute(text("PRAGMA table_info('snippet_usage')")).fetchall()
        existing_cols = {row[1] for row in col_info}  # PRAGMA returns (cid,name,...) at indexes

        for i, snippet in enumerate(request.snippets):
            # Build insert data according to existing columns
            insert_cols = []
            insert_vals = {}

            # Always include id and session_id and snippet_id if present
            if 'id' in existing_cols:
                insert_cols.append('id')
                insert_vals['id'] = uuid.uuid4().hex
            if 'session_id' in existing_cols:
                insert_cols.append('session_id')
                insert_vals['session_id'] = str(db_session.id)
            if 'user_id' in existing_cols:
                insert_cols.append('user_id')
                insert_vals['user_id'] = str(db_session.user_id) if db_session.user_id else None
            if 'snippet_id' in existing_cols:
                insert_cols.append('snippet_id')
                insert_vals['snippet_id'] = str(snippet.snippet_id)
            # Map started_at/completed_at or presented_at
            if 'started_at' in existing_cols:
                insert_cols.append('started_at')
                insert_vals['started_at'] = snippet.started_at
            if 'completed_at' in existing_cols:
                insert_cols.append('completed_at')
                insert_vals['completed_at'] = snippet.completed_at
            if 'presented_at' in existing_cols and 'started_at' not in existing_cols:
                insert_cols.append('presented_at')
                insert_vals['presented_at'] = snippet.started_at
            if 'user_wpm' in existing_cols:
                insert_cols.append('user_wpm')
                insert_vals['user_wpm'] = snippet.wpm
            if 'user_accuracy' in existing_cols:
                insert_cols.append('user_accuracy')
                insert_vals['user_accuracy'] = snippet.accuracy
            if 'snippet_position' in existing_cols:
                insert_cols.append('snippet_position')
                insert_vals['snippet_position'] = i
            if 'difficulty_snapshot' in existing_cols:
                insert_cols.append('difficulty_snapshot')
                insert_vals['difficulty_snapshot'] = snippet.difficulty

            if insert_cols:
                cols_sql = ", ".join(insert_cols)
                vals_sql = ", ".join(f":{c}" for c in insert_cols)
                sql = text(f"INSERT INTO snippet_usage ({cols_sql}) VALUES ({vals_sql})")
                db.execute(sql, insert_vals)
            
        db.commit()
        db.refresh(db_session)
        logger.info(f"Saved session {db_session.id} with flow_score {flow_score}")
        return db_session
    except Exception as e:
        db.rollback()
        logger.exception("Failed to save session")
        # Return a controlled HTTP error so CORS middleware can add headers
        raise HTTPException(status_code=500, detail="Failed to save session")
