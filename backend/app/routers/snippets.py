from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import SnippetRetrieveRequest, SnippetResponse
from app.models.db_models import User, Snippet
from app.ml.lints_agent import agent
from app.ml.user_features import UserFeatureExtractor
from typing import List, Dict, Any, cast
import time
import uuid
import numpy as np

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/retrieve")
def retrieve_snippets(
    req: Request, request: SnippetRetrieveRequest, db: Session = Depends(get_db)
):
    """
    Retrieves and ranks snippets based on user state using LinTS Agent.
    """

    # helper to compute rolling WPM
    def compute_rolling_wpm(
        timestamps: List[float], windows: List[int] = [15, 30, 45, 60]
    ) -> Dict[str, float]:
        now = time.time()
        wpm_map: Dict[str, float] = {}
        for w in windows:
            cutoff = now - w
            chars = sum(1 for t in timestamps if t >= cutoff)
            wpm = (chars * 60) / (5 * max(1, w))
            wpm_map[str(w)] = round(wpm, 2)
        return wpm_map

    # 1. Fetch User & Compute Context
    user_ema: List[float] = []
    user_std: List[float] = []  # Placeholder
    prev_embedding: List[float] | None = None

    if request.user_state.user_id:
        try:
            # Handle user_id as string or UUID
            uid_str = str(request.user_state.user_id)
            user = db.query(User).filter(User.id == uid_str).first()

            # Auto-create anonymous user if doesn't exist
            if user is None:
                user = User(id=uid_str, is_anonymous=True)
                db.add(user)
                db.flush()

            features_obj: Dict[str, Any] = cast(Dict[str, Any], user.features or {})
            if features_obj:
                extractor = UserFeatureExtractor.from_dict(features_obj)
                user_ema = extractor.compute_user_features().tolist()

                # Optional: Compute STD from short_term_history if available
                # if extractor.short_term_history:
                #    user_std = np.std(extractor.short_term_history, axis=0).tolist()
        except Exception as e:
            print(f"Error fetching user context: {e}")

    # 2. Get Previous Snippet Embedding
    if request.current_snippet_id:
        try:
            prev_snip = (
                db.query(Snippet)
                .filter(Snippet.id == request.current_snippet_id)
                .first()
            )
            if prev_snip is not None:
                emb_val = cast(List[float] | None, prev_snip.embedding)
                if emb_val is not None:
                    prev_embedding = emb_val
        except Exception as e:
            print(f"Error fetching prev snippet: {e}")

    # 3. Agent Prediction
    user_context = agent.get_context(user_ema, user_std, prev_embedding)
    query_vector = agent.predict(user_context)

    # 4. Vector Search
    vector_store = req.app.state.vector_store
    candidates = vector_store.search(
        query_vector=np.array(query_vector, dtype=np.float32), k=50
    )

    if not candidates:
        # Fallback if no candidates found (e.g., empty index)
        # Search with random vector or raise
        candidates = vector_store.search(
            query_vector=np.random.randn(16).astype(np.float32), k=50
        )
        if not candidates:
            raise HTTPException(status_code=404, detail="No snippets available.")

    # 5. Filter out recently shown snippets
    recent_ids = request.user_state.recentSnippetIds or []
    current_id = request.current_snippet_id
    exclude_ids = set(str(sid) for sid in recent_ids)
    if current_id:
        exclude_ids.add(str(current_id))

    filtered_snippets = [s for s in candidates if str(s.get("id")) not in exclude_ids]

    # If we filtered everything, just use the top candidate
    if not filtered_snippets and candidates:
        filtered_snippets = candidates[:1]

    # 6. Select Snippet Probabilistically Based on Distance
    # Convert L2 distances to probabilities using softmax with temperature
    # Lower distance = higher probability
    distances = np.array([s.get("distance", 0.0) for s in filtered_snippets])
    
    # Use negative distances (closer = better) and apply softmax with temperature
    # Temperature controls exploration: lower temp = more exploitation, higher = more exploration
    temperature = 2.0
    neg_distances = -distances / temperature
    
    # Softmax to get probabilities
    exp_scores = np.exp(neg_distances - np.max(neg_distances))  # Subtract max for numerical stability
    probabilities = exp_scores / np.sum(exp_scores)
    
    # Sample according to probabilities
    selected_idx = np.random.choice(len(filtered_snippets), p=probabilities)
    top_snippet_data = filtered_snippets[selected_idx]

    top_snippet = {
        "id": top_snippet_data.get("id"),
        "words": top_snippet_data.get("words"),
    }

    # 7. Compute rolling WPM
    base_wpm = 0.0
    timestamps = getattr(request.user_state, "keystroke_timestamps", None)
    if timestamps:
        wpm_windows = compute_rolling_wpm(timestamps)
        base_wpm = float(wpm_windows.get("60", 0.0))
    else:
        base_wpm = float(getattr(request.user_state, "rollingWpm", 0.0))
        wpm_windows = {str(w): round(base_wpm, 2) for w in [15, 30, 45, 60]}

    # 8. Compute predicted performance metrics from user features
    predicted_wpm = None
    predicted_accuracy = None
    predicted_consistency = None

    if len(user_ema) >= 57:  # Ensure we have valid user features
        # Use existing feature indices from user features
        predicted_wpm = (
            user_ema[21] if len(user_ema) > 21 else base_wpm
        )  # IDX_WPM_EFFECTIVE
        predicted_accuracy = user_ema[0] if len(user_ema) > 0 else 0.95  # IDX_ACCURACY
        predicted_consistency = (
            1.0 / (1.0 + user_ema[10]) if len(user_ema) > 10 else 0.8
        )  # Inverse of IKI_CV

    return {
        "snippet": top_snippet,
        "wpm_windows": wpm_windows,
        "predicted_wpm": predicted_wpm,
        "predicted_accuracy": predicted_accuracy,
        "predicted_consistency": predicted_consistency,
    }
