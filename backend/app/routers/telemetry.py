from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import SnippetLog, UserState
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


@router.post("/snippet")
async def receive_snippet_telemetry(payload: dict, db: Session = Depends(get_db)):
    """Accept per-snippet telemetry for online tuning and offline training.

    Payload shape:
    {
      "snippet": { ... SnippetLog ... },
      "user_state": { ... optional UserState ... }
    }
    """
    try:
        snippet = payload.get('snippet')
        user_state = payload.get('user_state')

        # Defensive: check required fields
        if not snippet or 'snippet_id' not in snippet:
            raise HTTPException(status_code=400, detail='Missing snippet payload')

        # Insert into snippet_usage table using available columns
        col_info = db.execute(text("PRAGMA table_info('snippet_usage')")).fetchall()
        existing_cols = {row[1] for row in col_info}

        insert_cols = []
        insert_vals = {}
        if 'id' in existing_cols:
            insert_cols.append('id')
            insert_vals['id'] = uuid.uuid4().hex
        if 'session_id' in existing_cols:
            # no session id here
            insert_cols.append('session_id')
            insert_vals['session_id'] = None
        if 'user_id' in existing_cols:
            insert_cols.append('user_id')
            insert_vals['user_id'] = None
        if 'snippet_id' in existing_cols:
            insert_cols.append('snippet_id')
            insert_vals['snippet_id'] = snippet.get('snippet_id')
        if 'started_at' in existing_cols:
            insert_cols.append('started_at')
            insert_vals['started_at'] = snippet.get('started_at')
        if 'completed_at' in existing_cols:
            insert_cols.append('completed_at')
            insert_vals['completed_at'] = snippet.get('completed_at')
        if 'presented_at' in existing_cols and 'started_at' not in existing_cols:
            insert_cols.append('presented_at')
            insert_vals['presented_at'] = snippet.get('started_at')
        if 'user_wpm' in existing_cols:
            insert_cols.append('user_wpm')
            insert_vals['user_wpm'] = snippet.get('wpm')
        if 'user_accuracy' in existing_cols:
            insert_cols.append('user_accuracy')
            insert_vals['user_accuracy'] = snippet.get('accuracy')
        if 'snippet_position' in existing_cols:
            insert_cols.append('snippet_position')
            insert_vals['snippet_position'] = snippet.get('position', 0)
        if 'difficulty_snapshot' in existing_cols:
            insert_cols.append('difficulty_snapshot')
            insert_vals['difficulty_snapshot'] = snippet.get('difficulty')

        if insert_cols:
            cols_sql = ", ".join(insert_cols)
            vals_sql = ", ".join(f":{c}" for c in insert_cols)
            sql = text(f"INSERT INTO snippet_usage ({cols_sql}) VALUES ({vals_sql})")
            db.execute(sql, insert_vals)
            db.commit()

        # Optionally, we could also enqueue the telemetry for async processing
        logger.info(f"Received telemetry for snippet {snippet.get('snippet_id')}")
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception('Failed to process telemetry')
        raise HTTPException(status_code=500, detail='Failed to process telemetry')
