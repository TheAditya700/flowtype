# ----------------------------
# telemetry.py  (rewritten)
# ----------------------------
from fastapi import APIRouter, HTTPException, Request
import logging, os, redis
from rq import Queue
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.db_models import TelemetrySnippetRaw
from app.tasks.telemetry_tasks import process_telemetry

router = APIRouter()
logger = logging.getLogger(__name__)


def get_redis_queue():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    r = redis.from_url(redis_url)
    return Queue("telemetry", connection=r)


@router.post("/snippet")
async def receive_snippet_telemetry(payload: dict, request: Request):
    """
    Saves snippet-level online telemetry:
        - latency
        - error streaks
        - hesitation patterns
        - key dynamics
        - snippet difficulty
        - model score
        - reward contribution
        
    This allows:
        → Improving Two-Tower scorer
        → RL reward-tuning  
        → GRU-based user modeling
    """

    snippet = payload.get("snippet")
    if not snippet:
        raise HTTPException(status_code=400, detail="Missing snippet payload")

    try:
        # ----------------------------------------------
        # Store durable raw telemetry
        # ----------------------------------------------
        db: Session = SessionLocal()
        raw = TelemetrySnippetRaw(
            payload=payload,
            user_id=payload.get("user_state", {}).get("user_id"),
            snippet_id=snippet.get("snippet_id"),
            model_score=snippet.get("model_score"),
            reward_estimate=payload.get("reward_estimate"),
            source=(request.client.host if request.client else None)
        )
        db.add(raw)
        db.commit()
        raw_id = str(raw.id)

    except Exception as e:
        logger.exception("Telemetry write failure")
        raise HTTPException(status_code=500, detail="Failed to save telemetry")
    finally:
        db.close()

    # -------------------------------------------------
    # Best-effort enqueue for processing (async)
    # -------------------------------------------------
    enqueued, job_id = False, None

    try:
        q = get_redis_queue()
        job = q.enqueue(process_telemetry, payload)
        enqueued, job_id = True, job.id
        logger.info("Telemetry job %s queued", job.id)
    except Exception:
        logger.warning("Redis unavailable; continuing without async job")

    return {
        "status": "ok",
        "raw_id": raw_id,
        "enqueued": enqueued,
        "job_id": job_id,
    }
