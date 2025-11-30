from app.database import SessionLocal
from app.models.db_models import TelemetrySnippetRaw
from sqlalchemy.exc import SQLAlchemyError
import logging
import uuid

logger = logging.getLogger(__name__)


def process_telemetry(payload: dict, source: str | None = None):
    """Background job to persist raw telemetry payloads into DB.

    This function is intended to be enqueued by RQ and executed by a worker.
    """
    db = SessionLocal()
    try:
        raw = TelemetrySnippetRaw(
            id=uuid.uuid4(),
            payload=payload,
            user_id=payload.get('user_state', {}).get('user_id') if payload.get('user_state') else None,
            session_id=payload.get('snippet', {}).get('session_id') if payload.get('snippet') else None,
            source=source,
        )
        db.add(raw)
        db.commit()
        logger.info('Persisted raw telemetry id=%s', raw.id)
        return str(raw.id)
    except SQLAlchemyError:
        logger.exception('Failed to persist raw telemetry')
        db.rollback()
        raise
    finally:
        db.close()
