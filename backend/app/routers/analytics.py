from fastapi import APIRouter, HTTPException
from app.models.schema import AnalyticsRequest, AnalyticsResponse, SpeedPoint, ReplayEvent
from app.ml.user_features import UserFeatureExtractor
import numpy as np

router = APIRouter()

@router.post("/calculate", response_model=AnalyticsResponse)
def calculate_metrics(request: AnalyticsRequest):
    """
    Calculates detailed metrics for a single session using UserFeatureExtractor logic.
    Returns flow/fluency scores.
    """
    try:
        # 1. Instantiate a fresh extractor
        extractor = UserFeatureExtractor()
        
        # 2. Convert KeystrokeEvent list to dict format expected by extractor
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
        
        session_data = {
            'keystroke_events': events_dicts,
            'wpm': request.wpm,
            'accuracy': request.accuracy
        }
        
        # 3. Update extractor with this session
        extractor.update_from_session(session_data)
        
        # 4. Extract stats directly from extractor internals or via compute_user_features
        # We access internal stats for specific dashboard metrics
        
        # Helpers
        def safe_div(n, d, default=0.0):
            return n / d if d > 0 else default
            
        def get_iki_cv(key):
            s = extractor.iki_stats[key]
            if s['count'] == 0: return 0.0
            mean = s['sum'] / s['count']
            var = (s['sum_sq'] / s['count']) - (mean ** 2)
            std = np.sqrt(max(0, var))
            return safe_div(std, mean)

        # Fluency ~ 1 - CV (Coefficient of Variation)
        # CV is usually around 0.3 - 0.8. We map 0.0 -> 100, 1.0 -> 0 roughly.
        def cv_to_score(cv):
            return max(0.0, min(100.0, (1.0 - cv) * 100))
            
        cv_global = get_iki_cv('global')
        cv_L2L = get_iki_cv('L2L')
        cv_R2R = get_iki_cv('R2R')
        cv_cross = get_iki_cv('cross')
        
        # Rollover Rate
        total_transitions = extractor.iki_stats['global']['count']
        rollover_rate = safe_div(extractor.rollover_count, total_transitions)
        
        # Detailed Stats Calculation
        avgIki = extractor.iki_stats['global']['sum'] / extractor.iki_stats['global']['count'] if extractor.iki_stats['global']['count'] > 0 else 0
        
        # KSPC approximation: total_presses / (total_presses - backspaces - errors) 
        # (Using logic from compute_user_features roughly)
        correct_presses = extractor.total_presses - extractor.backspace_count - extractor.total_errors
        kspc = safe_div(extractor.total_presses, correct_presses, default=1.0)
        
        # Heatmap Data Generation
        # Iterate events again to get per-key IKI which extractor doesn't store in char_stats
        char_iki_stats = {} # char -> {total_iki, count}
        
        sorted_events = sorted(events_dicts, key=lambda x: x['timestamp'])
        
        # --- Speed Series Calculation ---
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
            
            # Convert buckets to list
            raw_points = []
            sorted_times = sorted(buckets.keys())
            for t in sorted_times:
                b = buckets[t]
                # Raw WPM = (chars / 5) * 60 (since bucket is 1 sec)
                raw_wpm = (b['chars'] / 5.0) * 60.0
                raw_points.append({
                    'time': t / 1000.0,
                    'rawWpm': raw_wpm,
                    'errors': b['errors']
                })
                
            # Smoothing (Moving Average 3-window)
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

        # --- Replay / Chunking Calculation ---
        replay_events = []
        valid_events = [e for e in sorted_events if not e['isBackspace'] and e['isCorrect'] and len(e['key']) == 1]
        
        if valid_events:
            # Calculate IKIs for valid events only
            ikis = []
            rollovers = []
            
            for i in range(len(valid_events)):
                if i == 0:
                    ikis.append(0)
                    rollovers.append(False)
                else:
                    ikis.append(valid_events[i]['timestamp'] - valid_events[i-1]['timestamp'])
                    
                    # Rollover check: current down < prev up
                    prev_keyup = valid_events[i-1].get('keyup_timestamp')
                    curr_down = valid_events[i]['timestamp']
                    is_rollover = False
                    if prev_keyup and curr_down < prev_keyup:
                        is_rollover = True
                    rollovers.append(is_rollover)
            
            # Threshold
            if ikis:
                median_iki = float(np.median(ikis))
                threshold = median_iki * 1.8 if median_iki > 0 else 300
            else:
                threshold = 300
            
            for i, e in enumerate(valid_events):
                iki = ikis[i]
                is_rollover = rollovers[i]
                
                # Assign Snippet Index
                s_idx = 0
                if request.snippetBoundaries:
                    ts = e['timestamp']
                    for idx, boundary in enumerate(request.snippetBoundaries):
                        # Use loose boundary: if timestamp <= endTime
                        if ts <= boundary.endTime:
                            s_idx = idx
                            break
                    else:
                        # If beyond last boundary (margin of error), assign to last
                        s_idx = len(request.snippetBoundaries) - 1
                
                replay_events.append(ReplayEvent(
                    char=e['key'],
                    iki=iki,
                    isChunkStart=(iki > threshold),
                    isError=False,
                    snippetIndex=s_idx,
                    isRollover=is_rollover
                ))

        # Calculate avgChunkLength
        chunk_starts = sum(1 for e in replay_events if e.isChunkStart)
        num_chunks = 1 + chunk_starts
        avg_chunk_length = len(replay_events) / num_chunks if replay_events else 0.0

        # --- Heatmap IKI gathering ---
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
        for char, stats in extractor.char_stats.items():
            if stats['presses'] > 0:
                acc = 1.0 - (stats['errors'] / stats['presses'])
                
                # Calculate speed score
                avg_key_iki = 0
                if char in char_iki_stats and char_iki_stats[char]['count'] > 0:
                    avg_key_iki = char_iki_stats[char]['sum'] / char_iki_stats[char]['count']
                
                # Normalize speed (lower IKI is better). Map 50ms -> 1.0, 300ms -> 0.0
                speed = max(0.0, min(1.0, 1.0 - (avg_key_iki - 50) / 250))
                
                heatmap_data[char] = {
                    "accuracy": acc,
                    "speed": speed
                }

        return AnalyticsResponse(
            smoothness=cv_to_score(cv_global),
            rollover=rollover_rate * 100.0, # Percentage
            leftFluency=cv_to_score(cv_L2L),
            rightFluency=cv_to_score(cv_R2R),
            crossFluency=cv_to_score(cv_cross),
            speed=request.wpm,
            accuracy=request.accuracy,
            avgIki=avgIki,
            kspc=kspc,
            errors=extractor.total_errors,
            heatmapData=heatmap_data,
            speedSeries=speed_series,
            replayEvents=replay_events,
            avgChunkLength=avg_chunk_length
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
