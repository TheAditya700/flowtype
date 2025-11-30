from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import SnippetRetrieveRequest, SnippetResponse
from app.ml.user_encoder import get_user_embedding
from app.ml.ranker import rank_snippets
from typing import List

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/retrieve", response_model=List[SnippetResponse])
def retrieve_snippets(
    req: Request,
    request: SnippetRetrieveRequest, 
    db: Session = Depends(get_db)
):
    """
    Retrieves and ranks snippets based on user state.
    """
    # 1. Encode user state
    user_embedding = get_user_embedding(request.user_state)
    
    # 2. Access the vector store from the app state and search
    vector_store = req.app.state.vector_store
    candidate_snippets = vector_store.search(
        query_vector=user_embedding,
        k=20, # Retrieve 20 candidates
        difficulty_min=request.user_state.currentDifficulty - 1.5,
        difficulty_max=request.user_state.currentDifficulty + 1.5
    )

    # 3. Rank snippets
    ranked_snippets = rank_snippets(user_embedding, candidate_snippets)
    
    if not ranked_snippets:
        raise HTTPException(status_code=404, detail="No suitable snippets found.")
        
    # 4. Format and return top 5
    return [SnippetResponse(id=s['id'], words=s['words'], difficulty=s['difficulty']) for s in ranked_snippets[:5]]
