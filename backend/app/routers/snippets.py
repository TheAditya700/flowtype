from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import SnippetRetrieveRequest, SnippetResponse
from app.models.db_models import User
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

    # 1. Fetch User Stats from DB (Long Term Context)
    user_features = {}
    if request.user_state.user_id:
        user = db.query(User).filter(User.id == request.user_state.user_id).first()
        if user:
            user_features = user.features or {}

    # 2. Encode user state
    user_embedding = get_user_embedding(request.user_state, user_features_dict=user_features)
    
    # 3. Access the vector store from the app state and search
    vector_store = req.app.state.vector_store
    
    # Pure embedding search (difficulty filtering removed)
    current_diff = getattr(request.user_state, "currentDifficulty", 5.0)
    
    candidate_snippets = vector_store.search(
        query_vector=user_embedding,
        k=200,
    )

    # 4. Rank snippets
    ranked_snippets = rank_snippets(user_embedding, candidate_snippets, current_diff)
    
    if not ranked_snippets:
        raise HTTPException(status_code=404, detail="No suitable snippets found.")
    
    # 4.5. Filter out recently shown snippets and current snippet to avoid repetition
    recent_ids = getattr(request.user_state, "recentSnippetIds", None) or []
    current_id = request.current_snippet_id
    
    # Build exclude set from recent IDs + current snippet ID
    exclude_ids = set(str(sid) for sid in recent_ids)
    if current_id:
        exclude_ids.add(str(current_id))
    
    filtered_snippets = [s for s in ranked_snippets if str(s.get("id")) not in exclude_ids]
    
    # 5. Format top snippet with Exploration (Weighted Random)
    top_snippet = None
    if filtered_snippets:
        # Exploration: Take top K (e.g., 50) and sample
        top_k = filtered_snippets[:50]
        
        # Simple weighted probability: higher rank = higher chance
        # Weights: [50, 49, 48, ..., 1]
        weights = list(range(len(top_k), 0, -1))
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Pick one
        import numpy as np
        selected_idx = np.random.choice(len(top_k), p=probs)
        s = top_k[selected_idx]
        
        top_snippet = {"id": s.get("id"), "words": s.get("words"), "difficulty": s.get("difficulty")}

    # 6. Compute rolling WPM windows from optional keystroke timestamps
    timestamps = getattr(request.user_state, "keystroke_timestamps", None)
    if timestamps:
        wpm_windows = compute_rolling_wpm(timestamps)
    else:
        # fallback to provided rollingWpm (single value) if available
        base_wpm = getattr(request.user_state, "rollingWpm", 0.0)
        wpm_windows = {str(w): round(base_wpm, 2) for w in [15, 30, 45, 60]}

    return {"snippet": top_snippet, "wpm_windows": wpm_windows}

