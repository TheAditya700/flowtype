#!/usr/bin/env python3
"""
Prepare training batches from raw telemetry.

This script reads raw telemetry from `telemetry_snippet_raw`, extracts features,
and produces training examples. Output is written to `data/training_batches.json` or CSV.

Usage:
    python scripts/prepare_telemetry_batches.py [--output-format json|csv]
"""
import sys
import json
import csv
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal
from app.models.db_models import TelemetrySnippetRaw, Snippet
from sqlalchemy import func


def extract_features(raw_record: dict) -> dict:
    """
    Extract features from a raw telemetry payload.
    
    Returns a dict with:
    - snippet_id, session_id, user_id
    - wpm, accuracy, position, difficulty
    - timestamp
    - and other fields useful for training
    """
    snippet = raw_record.get('snippet', {})
    user_state = raw_record.get('user_state', {})
    
    return {
        'snippet_id': snippet.get('snippet_id'),
        'session_id': snippet.get('session_id'),
        'user_id': user_state.get('user_id'),
        'wpm': snippet.get('wpm'),
        'accuracy': snippet.get('accuracy'),
        'position': snippet.get('position', 0),
        'difficulty': snippet.get('difficulty'),
        'started_at': snippet.get('started_at'),
        'completed_at': snippet.get('completed_at'),
    }


def prepare_batches(output_format: str = 'json'):
    """
    Read raw telemetry and prepare training batches.
    
    Args:
        output_format: 'json' or 'csv'
    """
    db = SessionLocal()
    try:
        # Fetch all raw telemetry records (in production, consider pagination/date range)
        raw_records = db.query(TelemetrySnippetRaw).all()
        
        if not raw_records:
            print("No raw telemetry records found.")
            return
        
        examples = []
        for raw in raw_records:
            try:
                features = extract_features(raw.payload)
                # Skip incomplete records
                if not features['wpm'] or not features['accuracy']:
                    continue
                examples.append(features)
            except Exception as e:
                print(f"Skipping record {raw.id}: {e}")
                continue
        
        print(f"Extracted {len(examples)} training examples from {len(raw_records)} raw records.")
        
        # Write output
        output_dir = Path(__file__).parent.parent / 'data'
        output_dir.mkdir(exist_ok=True)
        
        if output_format == 'json':
            output_file = output_dir / 'training_batches.json'
            with open(output_file, 'w') as f:
                json.dump(examples, f, indent=2)
            print(f"Wrote {len(examples)} examples to {output_file}")
        elif output_format == 'csv':
            output_file = output_dir / 'training_batches.csv'
            if examples:
                keys = examples[0].keys()
                with open(output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(examples)
                print(f"Wrote {len(examples)} examples to {output_file}")
        
        return examples
    finally:
        db.close()


if __name__ == '__main__':
    output_fmt = 'json'
    if len(sys.argv) > 1 and sys.argv[1] in ('csv', 'json'):
        output_fmt = sys.argv[1]
    
    prepare_batches(output_format=output_fmt)
