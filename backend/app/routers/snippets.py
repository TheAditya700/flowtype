from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import SnippetRetrieveRequest, SnippetResponse
from app.models.db_models import User
# from app.ml.user_encoder import get_user_embedding # REMOVED
from app.ml.inference import rank_snippets # UPDATED
from typing import List, Dict, Any
import time
import uuid 
import numpy as np # Added numpy

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
        try:
            user_uuid = uuid.UUID(request.user_state.user_id) 
            user = db.query(User).filter(User.id == user_uuid).first()
            if user:
                user_features = user.features or {}
        except ValueError:
            pass 

    # 2. Retrieve Candidates (Heuristic / Metadata Search)
    # Since we moved to manual features, we don't have a shared embedding space trained yet.
    # We will retrieve candidates based on difficulty proximity first, then rank with HTOM.
    vector_store = req.app.state.vector_store
    current_diff = getattr(request.user_state, "currentDifficulty", 5.0)
    
    # Hack: Search with a random vector to just get a spread of snippets
    # Ideally, we'd query by difficulty range, but FAISS is fast enough to just get K random-ish ones
    # or we can use a dummy vector.
    dummy_vec = np.random.rand(30).astype('float32') # Dimension 30 matches FAISS index
    
    candidate_snippets = vector_store.search(
        query_vector=dummy_vec,
        k=200,
    )

    # 3. Rank snippets using HTOM
    ranked_snippets = rank_snippets(
        user_state=request.user_state,
        user_features_dict=user_features,
        candidates=candidate_snippets,
        target_difficulty=current_diff
    )
    
    if not ranked_snippets:
        raise HTTPException(status_code=404, detail="No suitable snippets found.")
    
    # 4. Filter out recently shown snippets
    recent_ids = getattr(request.user_state, "recentSnippetIds", None) or []
    current_id = request.current_snippet_id
    
    exclude_ids = set(str(sid) for sid in recent_ids)
    if current_id:
        exclude_ids.add(str(current_id))
    
    filtered_snippets = [s for s in ranked_snippets if str(s.get("id")) not in exclude_ids]
    
    # 5. Format top snippet with Exploration
    top_snippet = None
    if filtered_snippets:
        top_k = filtered_snippets[:20] # Narrower top-k since ranking is better now
        
        weights = list(range(len(top_k), 0, -1))
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        selected_idx = np.random.choice(len(top_k), p=probs)
        s = top_k[selected_idx]
        
        top_snippet = {"id": s.get("id"), "words": s.get("words"), "difficulty": s.get("difficulty")}

    # 6. Compute rolling WPM
    timestamps = getattr(request.user_state, "keystroke_timestamps", None)
    if timestamps:
        wpm_windows = compute_rolling_wpm(timestamps)
    else:
        base_wpm = getattr(request.user_state, "rollingWpm", 0.0)
        wpm_windows = {str(w): round(base_wpm, 2) for w in [15, 30, 45, 60]}

    return {"snippet": top_snippet, "wpm_windows": wpm_windows}

