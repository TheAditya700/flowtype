from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.db_models import User
from pydantic import BaseModel
import numpy as np
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class MergeProfileRequest(BaseModel):
    anon_user_id: str
    auth_user_id: str


def merge_embeddings(anon_embedding: list, auth_embedding: list, anon_weight: float = 0.3) -> list:
    """
    Merge two embedding vectors using weighted average.
    Gives more weight to the authenticated user's existing data.
    """
    if not anon_embedding:
        return auth_embedding
    if not auth_embedding:
        return anon_embedding
    
    anon_arr = np.array(anon_embedding)
    auth_arr = np.array(auth_embedding)
    
    # Weighted average: 30% anonymous, 70% authenticated
    merged = (anon_weight * anon_arr + (1 - anon_weight) * auth_arr).tolist()
    return merged


def merge_features(anon_features: dict, auth_features: dict) -> dict:
    """
    Merge user features (EMA stats, extractor data, etc.)
    """
    if not anon_features:
        return auth_features
    if not auth_features:
        return anon_features
    
    merged = auth_features.copy()
    
    # Merge raw extractor features
    if 'raw' in anon_features and 'raw' in auth_features:
        anon_raw = anon_features['raw']
        auth_raw = auth_features['raw']
        
        # Add session counts
        if 'session_count' in anon_raw and 'session_count' in auth_raw:
            merged['raw']['session_count'] = anon_raw['session_count'] + auth_raw['session_count']
        
        # Merge character stats (combine presses and errors)
        if 'char_stats' in anon_raw and 'char_stats' in auth_raw:
            for char, stats in anon_raw['char_stats'].items():
                if char in auth_raw['char_stats']:
                    merged['raw']['char_stats'][char]['presses'] += stats['presses']
                    merged['raw']['char_stats'][char]['errors'] += stats['errors']
                else:
                    merged['raw']['char_stats'][char] = stats.copy()
    
    # Merge EMA features (use weighted average)
    if 'ema' in anon_features and 'ema' in auth_features:
        anon_ema = anon_features['ema']
        auth_ema = auth_features['ema']
        
        if 'ema_mean' in anon_ema and 'ema_mean' in auth_ema:
            merged['ema']['ema_mean'] = merge_embeddings(
                anon_ema['ema_mean'], 
                auth_ema['ema_mean'],
                anon_weight=0.3
            )
        
        if 'ema_var' in anon_ema and 'ema_var' in auth_ema:
            merged['ema']['ema_var'] = merge_embeddings(
                anon_ema['ema_var'], 
                auth_ema['ema_var'],
                anon_weight=0.3
            )
    
    return merged


def merge_best_wpms(anon_wpms: dict, auth_wpms: dict) -> dict:
    """
    Merge best WPM stats - keep the maximum for each duration
    """
    merged = auth_wpms.copy()
    
    for duration, wpm in anon_wpms.items():
        if duration in merged:
            merged[duration] = max(merged[duration], wpm)
        else:
            merged[duration] = wpm
    
    return merged


@router.post("/merge")
def merge_profiles(request: MergeProfileRequest, db: Session = Depends(get_db)):
    """
    Merge anonymous user profile into authenticated user profile.
    
    - Combines embeddings using weighted average
    - Merges character stats and session counts
    - Keeps best WPMs across both profiles
    - Marks anonymous profile as merged
    """
    try:
        # Fetch both users
        anon_user = db.query(User).filter(User.id == request.anon_user_id).first()
        auth_user = db.query(User).filter(User.id == request.auth_user_id).first()
        
        if not anon_user:
            raise HTTPException(status_code=404, detail="Anonymous user not found")
        if not auth_user:
            raise HTTPException(status_code=404, detail="Authenticated user not found")
        
        if not anon_user.is_anonymous:
            raise HTTPException(status_code=400, detail="Source user is not anonymous")
        
        logger.info(f"Merging profile {request.anon_user_id} into {request.auth_user_id}")
        
        # Merge features
        auth_user.features = merge_features(
            anon_user.features or {},
            auth_user.features or {}
        )
        
        # Merge best WPMs
        auth_user.best_wpms = merge_best_wpms(
            anon_user.best_wpms or {"15": 0.0, "30": 0.0, "60": 0.0, "120": 0.0},
            auth_user.best_wpms or {"15": 0.0, "30": 0.0, "60": 0.0, "120": 0.0}
        )
        
        # Mark anonymous profile as merged
        anon_user.merged_into = auth_user.id
        
        db.commit()
        
        logger.info(f"Successfully merged profile {request.anon_user_id} into {request.auth_user_id}")
        
        return {
            "success": True,
            "message": "Profile merged successfully",
            "merged_features": auth_user.features,
            "merged_best_wpms": auth_user.best_wpms
        }
    
    except Exception as e:
        db.rollback()
        logger.exception("Failed to merge profiles")
        raise HTTPException(status_code=500, detail=str(e))
