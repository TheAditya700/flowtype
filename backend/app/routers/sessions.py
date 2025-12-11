from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.schema import (
    SessionCreateRequest,
    SessionResponse,
    SpeedPoint,
    ReplayEvent,
    SnippetBoundary
)
from app.models.db_models import TypingSession, User, Snippet
from app.ml.user_features import UserFeatureExtractor
from app.ml.feature_aggregator import update_long_term_features
from app.ml.lints_agent import agent
from sqlalchemy.sql import func
import logging, uuid, json
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/", response_model=SessionResponse)
def create_session(request: SessionCreateRequest, db: Session = Depends(get_db)):
    """
    Handles the end of a snippet typing session:
      - Saves the final session row
      - Saves snippet usage rows
      - Saves keystrokes
      - Updates User long-term stats (via UserFeatureExtractor & Agent EMA)
      - Computes reward and Updates LinTS Agent
      - Updates Best WPMs
      - Returns detailed analytics
    """
    try:
        logger.info(f"Received {len(request.snippets)} snippets in session")
        for idx, snippet in enumerate(request.snippets):
            logger.info(f"Snippet {idx}: {snippet.snippet_id}, partial={snippet.is_partial}, completed_words={snippet.completed_words}/{snippet.total_words}")
        
        # -----------------------------------------------------
        # 1. Setup User & Extractor
        # -----------------------------------------------------
        user = None
        user_ema_vector = []
        
        if request.user_id:
            user = db.query(User).filter(User.id == request.user_id).first()
            if not user:
                user = User(id=request.user_id, is_anonymous=True)
                db.add(user)
                db.flush()

        # Load features (JSON)
        # Structure: { "raw": {...}, "ema": {...} } or legacy just {...}
        features_json = user.features if user and user.features else {}
        
        # Backward compatibility check
        if "raw" in features_json:
            raw_state = features_json.get("raw", {})
            ema_state = features_json.get("ema", {})
        else:
            raw_state = features_json # Assume legacy is just raw
            ema_state = {}

        # Load extractor from raw state
        extractor = UserFeatureExtractor.from_dict(raw_state)
        
        # Prepare session dict for extractor
        events_dicts = [
            {
                "key": k.key,
                "isBackspace": k.isBackspace,
                "isCorrect": k.isCorrect,
                "timestamp": k.timestamp,
                "keyup_timestamp": k.keyup_timestamp
            } 
            for k in request.keystrokeData
        ]
        
        # Calculate wpm, accuracy, errors from keystrokeData
        # Each KeystrokeEvent represents one keystroke (not double-counting keydown/keyup)
        
        # Use provided duration (for timed modes) or calculate from keystroke timestamps
        # Timed modes send the actual session duration (15s, 30s, etc.)
        # Free mode sends duration calculated from keystrokes
        duration_seconds = request.durationSeconds
        
        is_free_mode = request.sessionMode == 'free'

        # For WPM: count only correct, non-backspace keystrokes
        correct_keystrokes = sum(1 for k in request.keystrokeData if k.isCorrect and not k.isBackspace)
        
        # For raw WPM: count all non-backspace keystrokes
        total_non_backspace_keystrokes = sum(1 for k in request.keystrokeData if not k.isBackspace)
        
        # For accuracy: total keystrokes (including incorrect ones, but not backspaces)
        accuracy_denominator = total_non_backspace_keystrokes
        
        # Errors: incorrect keystrokes (not backspaces)
        errors = sum(1 for k in request.keystrokeData if not k.isCorrect and not k.isBackspace)
        
        # Calculate WPM: (correct keystrokes / 5) / (duration in minutes)
        duration_minutes = duration_seconds / 60 if duration_seconds > 0 else 0.001
        calculated_wpm = (correct_keystrokes / 5) / duration_minutes if duration_minutes > 0 else 0
        
        # Calculate Raw WPM: (all non-backspace keystrokes / 5) / (duration in minutes)
        calculated_raw_wpm = (total_non_backspace_keystrokes / 5) / duration_minutes if duration_minutes > 0 else 0
        
        # Calculate Accuracy: (correct keystrokes / total non-backspace keystrokes)
        calculated_accuracy = (correct_keystrokes / accuracy_denominator) if accuracy_denominator > 0 else 0
        
        session_data = {
            'keystroke_events': events_dicts,
            'wpm': calculated_wpm,
            'accuracy': calculated_accuracy,
            'snippet_difficulty': request.difficultyLevel,
            'completed': True,
            'quit_progress': 1.0
        }
        
        # --- Pre-update Context for Agent ---
        # We need the EMA vector *before* this session updates it (conceptually),
        # or *current* belief about user.
        if ema_state and 'ema_mean' in ema_state:
            pre_session_features = ema_state['ema_mean']
        else:
            # Fallback: compute from extractor history if available, or zeros
            pre_session_features = extractor.compute_user_features().tolist()
            
        # Update Extractor with *current* session (accumulates counts)
        extractor.update_from_session(session_data)
        
        # Calculate new short-term feature vector for this session
        session_features_vector = extractor.compute_user_features().tolist()
        
        # Update EMA State using feature_aggregator
        new_ema_state = update_long_term_features(ema_state, session_features_vector)
        
        # Update User in DB
        if user:
            user.features = {
                "raw": extractor.to_dict(),
                "ema": new_ema_state
            }
            user.last_active = func.now()
            
            # --- Update Best WPMs (timed modes only) ---
            if not is_free_mode:
                current_wpm = calculated_wpm
                duration = duration_seconds
                
                # Helper to update best if current is better within tolerance window
                # Tolerance: +/- 10% or absolute time window?
                # User request: "best wpms for 15 30 60 and 120"
                # We usually map actual duration to closest bucket.
                
                best_wpms = dict(user.best_wpms) if user.best_wpms else {"15": 0.0, "30": 0.0, "60": 0.0, "120": 0.0}
                
                # Define buckets
                buckets = [15, 30, 60, 120]
                
                # Find closest bucket
                closest_bucket = min(buckets, key=lambda x: abs(x - duration))
                
                # Check if duration is reasonably close (e.g. within 20% or +/- 5s)
                # If user types for 16s, it counts for 15s bucket.
                # If 45s, counts for 30s or 60s? Maybe 60?
                if abs(closest_bucket - duration) / closest_bucket < 0.25: # 25% tolerance
                    bucket_key = str(closest_bucket)
                    if current_wpm > best_wpms.get(bucket_key, 0.0):
                        best_wpms[bucket_key] = current_wpm
                        user.best_wpms = best_wpms
            
            db.add(user)

        # -----------------------------------------------------
        # 2. LinTS Agent Update
        # -----------------------------------------------------
        total_session_reward = 0.0
        
        snippet_ids = [s.snippet_id for s in request.snippets]
        db_snippets = db.query(Snippet).filter(Snippet.id.in_(snippet_ids)).all()
        snippet_map = {str(s.id): s for s in db_snippets}
        
        # Calculate global smoothness proxies for reward calculation
        def safe_div(n, d, default=0.0):
            return n / d if d > 0 else default
            
        def get_iki_cv(key):
            s = extractor.iki_stats[key]
            if s['count'] == 0: return 0.0
            mean = s['sum'] / s['count']
            var = (s['sum_sq'] / s['count']) - (mean ** 2)
            std = np.sqrt(max(0, var))
            return safe_div(std, mean)
            
        global_iki_cv = get_iki_cv('global')
        total_intervals = extractor.spike_count + extractor.flow_intervals
        global_spike_rate = safe_div(extractor.spike_count, total_intervals)
        
        for s_res in request.snippets:
            s_db = snippet_map.get(s_res.snippet_id)
            if s_db and s_db.embedding:
                metrics_now = {
                    'accuracy': s_res.accuracy,
                    'wpm': s_res.wpm,
                    'iki_cv': global_iki_cv,
                    'spike_rate': global_spike_rate
                }
                
                # Reward: compare current metrics against PRE-SESSION EMA baseline
                r = agent.calculate_reward(metrics_now, pre_session_features)
                total_session_reward += r
                
                # Agent Update
                # Context: EMA, STD, PrevSnippet
                # We need STD from the EMA state if available
                user_std = []
                if 'ema_var' in new_ema_state: # Use new variance or old? Usually old context.
                    # Using old var from ema_state (before update) if possible, but we updated it.
                    # Let's use new_ema_state var as proxy for "variance of user performance"
                    # Actually feature_aggregator returns 'ema_var'.
                    user_std = np.sqrt(np.maximum(new_ema_state['ema_var'], 0)).tolist()
                
                user_context = agent.get_context(
                    user_ema=pre_session_features,
                    user_std=user_std,
                    prev_snippet_embedding=None 
                )
                
                agent.update(user_context, s_db.embedding, r)
        
        agent.save()

        # -----------------------------------------------------
        # 3. Calculate Smoothness Score (needed for DB save)
        # -----------------------------------------------------
        # Smoothness: Agent-style weighted formula (matching lints_agent.py)
        # smoothness = 0.5 * (1 / (1 + iki_cv)) + 0.5 * (1 - spike_rate)
        agent_smoothness_value = 0.5 * (1.0 / (1.0 + global_iki_cv)) + 0.5 * (1.0 - global_spike_rate)
        # Convert to 0-100 scale for UI and ensure it's a Python float
        smoothness_score = float(agent_smoothness_value * 100)

        # -----------------------------------------------------
        # 4. Save Session to DB
        # -----------------------------------------------------
        # Extract snippet IDs and embeddings
        snippet_ids = [s.snippet_id for s in request.snippets]
        snippet_embeddings = []
        for sid in snippet_ids:
            snippet = snippet_map.get(sid)
            if snippet and snippet.processed_embedding:
                snippet_embeddings.append(snippet.processed_embedding)
            else:
                snippet_embeddings.append(None)
        
        # Get user embedding (130-dim state vector)
        user_embedding_vec = None
        if user_ema_vector:  # This is the EMA mean vector from earlier
            user_embedding_vec = user_ema_vector  # Already constructed as [ema | std | prev_snippet]
        
        db_session = TypingSession(
            user_id=request.user_id,
            duration_seconds=float(duration_seconds),
            created_at=func.now(),
            user_embedding=user_embedding_vec,
            snippet_ids=snippet_ids,
            snippet_embeddings=snippet_embeddings,
            keystroke_events=events_dicts,
            actual_wpm=float(calculated_wpm),
            actual_accuracy=float(calculated_accuracy),
            actual_consistency=float(smoothness_score),
            predicted_wpm=float(request.predicted_wpm) if request.predicted_wpm is not None else None,
            predicted_accuracy=float(request.predicted_accuracy) if request.predicted_accuracy is not None else None,
            predicted_consistency=float(request.predicted_consistency) if request.predicted_consistency is not None else None,
            errors=int(errors),
            raw_wpm=float(calculated_raw_wpm),
            reward=float(total_session_reward),
        )
        db.add(db_session)

        db.commit()
        db.refresh(db_session)

        # -----------------------------------------------------
        # 5. Generate Detailed Analytics (Response)
        # -----------------------------------------------------
        
        # Helpers for Analytics
        def cv_to_score(cv):
            return max(0.0, min(100.0, (1.0 - cv) * 100))
            
        cv_global = get_iki_cv('global')
        cv_L2L = get_iki_cv('L2L')
        cv_R2R = get_iki_cv('R2R')
        cv_cross = get_iki_cv('cross')
        
        # Rollover Rate (Overall & Per-Hand)
        total_transitions = extractor.iki_stats['global']['count']
        rollover_rate = safe_div(extractor.rollover_count, total_transitions)
        
        # Hand-specific rollover rates
        l2l_transitions = extractor.trans_stats['L2L']['presses']
        r2r_transitions = extractor.trans_stats['R2R']['presses']
        cross_transitions = extractor.trans_stats['cross']['presses']
        
        rollover_l2l_rate = safe_div(extractor.roll_trans_counts['L2L'], l2l_transitions)
        rollover_r2r_rate = safe_div(extractor.roll_trans_counts['R2R'], r2r_transitions)
        rollover_cross_rate = safe_div(extractor.roll_trans_counts['cross'], cross_transitions)
        
        # Avg IKI
        avgIki = safe_div(extractor.iki_stats['global']['sum'], extractor.iki_stats['global']['count'])
        
        # KSPC
        correct_presses = extractor.total_presses - extractor.backspace_count - extractor.total_errors
        kspc = safe_div(extractor.total_presses, correct_presses, default=1.0)
        
        # Heatmap Data & Speed Series
        char_iki_stats = {} 
        sorted_events = sorted(events_dicts, key=lambda x: x['timestamp'])
        
        # Speed Series
        speed_series = []
        if sorted_events:
            bucket_size = 1000 # 1 second
            start_time = sorted_events[0]['timestamp']
            duration_ms = sorted_events[-1]['timestamp'] - start_time
            max_time = int(np.ceil(duration_ms / bucket_size) * bucket_size)
            
            buckets = {t: {'chars': 0, 'errors': 0} for t in range(0, max_time + bucket_size, bucket_size)}
            
            for e in sorted_events:
                rel_time = e['timestamp'] - start_time
                bucket_time = (rel_time // bucket_size) * bucket_size
                if bucket_time in buckets:
                    if not e['isBackspace']:
                        buckets[bucket_time]['chars'] += 1
                    if not e['isCorrect']:
                        buckets[bucket_time]['errors'] += 1
            
            raw_points = []
            sorted_times = sorted(buckets.keys())
            for t in sorted_times:
                b = buckets[t]
                raw_wpm = (b['chars'] / 5.0) * 60.0
                raw_points.append({
                    'time': t / 1000.0,
                    'rawWpm': raw_wpm,
                    'errors': b['errors']
                })
                
            # Smoothing
            for i in range(len(raw_points)):
                start = max(0, i - 2)
                window = raw_points[start : i+1]
                avg_wpm = sum(p['rawWpm'] for p in window) / len(window)
                speed_series.append(SpeedPoint(
                    time=raw_points[i]['time'],
                    wpm=round(avg_wpm),
                    rawWpm=raw_points[i]['rawWpm'],
                    errors=raw_points[i]['errors']
                ))

        # Replay Events
        replay_events = []
        valid_events = [e for e in sorted_events if not e['isBackspace'] and e['isCorrect'] and len(e['key']) == 1]
        
        if valid_events:
            ikis = []
            rollovers = []
            for i in range(len(valid_events)):
                if i == 0:
                    ikis.append(0)
                    rollovers.append(False)
                else:
                    ikis.append(valid_events[i]['timestamp'] - valid_events[i-1]['timestamp'])
                    prev_keyup = valid_events[i-1].get('keyup_timestamp')
                    curr_down = valid_events[i]['timestamp']
                    is_rollover = (prev_keyup and curr_down < prev_keyup)
                    rollovers.append(is_rollover)
            
            median_iki = float(np.median(ikis)) if ikis else 0
            threshold = median_iki * 1.8 if median_iki > 0 else 300
            
            snippet_boundaries = []
            for s in request.snippets:
                if s.started_at and s.completed_at:
                    snippet_boundaries.append(SnippetBoundary(startTime=s.started_at, endTime=s.completed_at))
            
            for i, e in enumerate(valid_events):
                iki = ikis[i]
                
                # Snippet Index
                s_idx = 0
                ts = e['timestamp']
                for idx, boundary in enumerate(snippet_boundaries):
                    if ts <= boundary.endTime:
                        s_idx = idx
                        break
                else:
                    if snippet_boundaries:
                        s_idx = len(snippet_boundaries) - 1
                
                replay_events.append(ReplayEvent(
                    char=e['key'],
                    iki=iki,
                    isChunkStart=(iki > threshold),
                    isError=False,
                    snippetIndex=s_idx,
                    isRollover=rollovers[i]
                ))

        chunk_starts = sum(1 for e in replay_events if e.isChunkStart)
        avg_chunk_length = len(replay_events) / (1 + chunk_starts) if replay_events else 0.0

        # Build heatmap data from CURRENT SESSION only
        session_char_stats = {}  # Character stats for this session only
        for event in request.keystrokeData:
            if event.isBackspace:
                continue
            char = event.key.lower()
            if char not in session_char_stats:
                session_char_stats[char] = {'presses': 0, 'errors': 0}
            session_char_stats[char]['presses'] += 1
            if not event.isCorrect:
                session_char_stats[char]['errors'] += 1
        
        # Heatmap IKI - calculate per-character inter-keystroke intervals
        for i in range(1, len(sorted_events)):
            curr = sorted_events[i]
            prev = sorted_events[i-1]
            if curr['timestamp'] > prev['timestamp']:
                iki = curr['timestamp'] - prev['timestamp']
                char = curr['key'].lower()
                if char not in char_iki_stats:
                    char_iki_stats[char] = {'sum': 0.0, 'count': 0}
                char_iki_stats[char]['sum'] += iki
                char_iki_stats[char]['count'] += 1
                
        heatmap_data = {}
        for char, stats in session_char_stats.items():
            if stats['presses'] > 0:
                acc = 1.0 - (stats['errors'] / stats['presses'])
                avg_key_iki = 0
                speed = 1.0  # Default to 1.0 if no IKI data
                if char in char_iki_stats and char_iki_stats[char]['count'] > 0:
                    avg_key_iki = char_iki_stats[char]['sum'] / char_iki_stats[char]['count']
                    speed = max(0.0, min(1.0, 1.0 - (avg_key_iki - 50) / 250))
                heatmap_data[char] = {"accuracy": float(acc), "speed": float(speed)}

        return SessionResponse(
            session_id=str(db_session.id),
            reward=total_session_reward,
            durationSeconds=duration_seconds,
            wpm=calculated_wpm,
            rawWpm=calculated_raw_wpm,
            accuracy=calculated_accuracy,
            errors=errors,
            smoothness=smoothness_score,
            rollover=rollover_rate * 100.0,
            leftFluency=cv_to_score(cv_L2L),
            rightFluency=cv_to_score(cv_R2R),
            crossFluency=cv_to_score(cv_cross),
            rolloverL2L=rollover_l2l_rate * 100.0,
            rolloverR2R=rollover_r2r_rate * 100.0,
            rolloverCross=rollover_cross_rate * 100.0,
            avgIki=avgIki,
            kspc=kspc,
            avgChunkLength=avg_chunk_length,
            heatmapData=heatmap_data,
            speedSeries=speed_series,
            replayEvents=replay_events,
            snippets=request.snippets
        )

    except Exception as e:
        db.rollback()
        logger.exception("Failed to save session")
        raise HTTPException(status_code=500, detail=str(e))