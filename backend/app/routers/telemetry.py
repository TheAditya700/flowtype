from fastapi import APIRouter, HTTPException, Request
import logging
import os
import redis
from rq import Queue
from app.tasks.telemetry_tasks import process_telemetry
from app.database import SessionLocal
from app.models.db_models import TelemetrySnippetRaw

router = APIRouter()
logger = logging.getLogger(__name__)


def get_redis_queue():
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    r = redis.from_url(redis_url)
    return Queue('telemetry', connection=r)


@router.post("/snippet")
async def receive_snippet_telemetry(payload: dict, request: Request):
    """Accept per-snippet telemetry for online tuning and offline training.

    Hybrid approach:
    - Persist a minimal raw record synchronously into `telemetry_snippet_raw` for durability.
    - Best-effort enqueue the payload to Redis/RQ for background enrichment/processing.
    """
    try:
        snippet = payload.get('snippet')
        if not snippet or 'snippet_id' not in snippet:
            raise HTTPException(status_code=400, detail='Missing snippet payload')

        source = request.client.host if request.client else None

        # Synchronously persist a minimal raw record for durability (hybrid approach)
        db = SessionLocal()
        try:
            raw = TelemetrySnippetRaw(
                payload=payload,
                user_id=(payload.get('user_state') or {}).get('user_id'),
                session_id=(payload.get('snippet') or {}).get('session_id'),
                source=source,
            )
            db.add(raw)
            db.flush()  # Flush to get the ID assigned without committing yet
            raw_id = str(raw.id)
            db.commit()
            logger.info('Persisted raw telemetry id=%s', raw_id)
        except Exception:
            db.rollback()
            logger.exception('Failed to persist raw telemetry synchronously')
            raise HTTPException(status_code=500, detail='Failed to persist telemetry')
        finally:
            db.close()

        # Try to enqueue for background enrichment; if Redis is unavailable, return ok
        enqueued = False
        job_id = None
        try:
            q = get_redis_queue()
            job = q.enqueue(process_telemetry, payload, source)
            enqueued = True
            job_id = job.id
            logger.info('Enqueued telemetry job %s for snippet %s', job.id, snippet.get('snippet_id'))
        except Exception:
            logger.exception('Redis enqueue failed; continuing without enqueue')

        return {"status": "ok", "raw_id": raw_id, "enqueued": enqueued, "job_id": job_id}
    except HTTPException:
        raise
    except Exception:
        logger.exception('Failed to handle telemetry request')
        raise HTTPException(status_code=500, detail='Failed to handle telemetry')
