from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import SnippetRetrieveRequest, SnippetResponse
from app.ml.user_encoder import get_user_embedding
from app.ml.ranker import rank_snippets
from typing import List, Dict, Any
import time

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/retrieve")
def retrieve_snippets(
    req: Request,
    request: SnippetRetrieveRequest, 
    db: Session = Depends(get_db)
):
    """
    Retrieves and ranks snippets based on user state and returns rolling WPM windows.

    The frontend may optionally include a list of keystroke timestamps in
    `request.user_state.keystroke_timestamps` (list of epoch-second floats). If
    present, rolling WPM for windows [15,30,45,60] seconds will be computed from
    those timestamps. Otherwise the endpoint will fall back to `user_state.rollingWpm`.
    """
    # helper to compute rolling WPM from timestamps
    def compute_rolling_wpm(timestamps: List[float], windows: List[int] = [15, 30, 45, 60]) -> Dict[str, float]:
        now = time.time()
        wpm_map: Dict[str, float] = {}
        for w in windows:
            cutoff = now - w
            chars = sum(1 for t in timestamps if t >= cutoff)
            # WPM = (chars / 5) / (minutes) = chars * 60 / (5 * seconds)
            wpm = (chars * 60) / (5 * max(1, w))
            wpm_map[str(w)] = round(wpm, 2)
        return wpm_map

    # 1. Encode user state
    user_embedding = get_user_embedding(request.user_state)
    
    # 2. Access the vector store from the app state and search
    vector_store = req.app.state.vector_store
    # Relaxed difficulty filtering to ensure we have enough candidates
    # Range: ±2.0 from current difficulty allows for progressive challenge
    current_diff = getattr(request.user_state, "currentDifficulty", 5.0)
    difficulty_min = max(1.0, current_diff - 2.0)
    difficulty_max = min(10.0, current_diff + 2.0)
    
    candidate_snippets = vector_store.search(
        query_vector=user_embedding,
        k=50,  # Retrieve more candidates for better filtering
        difficulty_min=difficulty_min,
        difficulty_max=difficulty_max
    )

    # 3. Rank snippets
    ranked_snippets = rank_snippets(user_embedding, candidate_snippets)
    
    if not ranked_snippets:
        raise HTTPException(status_code=404, detail="No suitable snippets found.")
    
    # 3.5. Filter out recently shown snippets and current snippet to avoid repetition
    recent_ids = getattr(request.user_state, "recentSnippetIds", None) or []
    current_id = request.current_snippet_id
    
    # Build exclude set from recent IDs + current snippet ID
    exclude_ids = set(str(sid) for sid in recent_ids)
    if current_id:
        exclude_ids.add(str(current_id))
    
    filtered_snippets = [s for s in ranked_snippets if str(s.get("id")) not in exclude_ids]
    
    # If all top snippets were filtered, fallback to showing them anyway (better UX than nothing)
    if not filtered_snippets:
        filtered_snippets = ranked_snippets
        
    # 4. Format top 1 snippet only
    top_snippet = None
    if filtered_snippets:
        s = filtered_snippets[0]
        top_snippet = {"id": s.get("id"), "words": s.get("words"), "difficulty": s.get("difficulty")}

    # 5. Compute rolling WPM windows from optional keystroke timestamps
    timestamps = getattr(request.user_state, "keystroke_timestamps", None)
    if timestamps:
        wpm_windows = compute_rolling_wpm(timestamps)
    else:
        # fallback to provided rollingWpm (single value) if available
        base_wpm = getattr(request.user_state, "rollingWpm", 0.0)
        wpm_windows = {str(w): round(base_wpm, 2) for w in [15, 30, 45, 60]}

    return {"snippet": top_snippet, "wpm_windows": wpm_windows}
