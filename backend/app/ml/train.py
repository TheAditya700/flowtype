#!/usr/bin/env python3
"""
Train a baseline ranker model on telemetry data.

This script reads prepared training batches and trains a simple LightGBM model
to predict user performance (WPM or accuracy) based on snippet features.

Usage:
    python app/ml/train.py [--model-type ranker|embedding] [--output-model app/ml/models/ranker.pkl]

Model types:
    - ranker: LightGBM regressor to predict WPM improvement
    - embedding: Placeholder for fine-tuned sentence-transformers (future)
"""
import sys
import json
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
except ImportError:
    print("Warning: lightgbm, numpy, or pandas not installed. Install via:")
    print("  pip install lightgbm numpy pandas")
    sys.exit(1)


def load_training_data(batch_file: str = None) -> pd.DataFrame:
    """Load training batches from JSON or CSV."""
    if not batch_file:
        batch_file = Path(__file__).parent.parent / 'data' / 'training_batches.json'
    
    batch_file = Path(batch_file)
    
    if not batch_file.exists():
        print(f"Training batches file not found: {batch_file}")
        print("Run: python scripts/prepare_telemetry_batches.py")
        return None
    
    if batch_file.suffix == '.json':
        with open(batch_file) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif batch_file.suffix == '.csv':
        return pd.read_csv(batch_file)
    else:
        raise ValueError(f"Unsupported file format: {batch_file.suffix}")


def train_ranker(df: pd.DataFrame, output_model: str = None):
    """Train a LightGBM ranker model."""
    if output_model is None:
        output_model = Path(__file__).parent / 'models' / 'ranker.pkl'
    
    output_model = Path(output_model)
    output_model.parent.mkdir(exist_ok=True)
    
    # Prepare features and target
    # Features: position, difficulty, accuracy (use WPM as target)
    feature_cols = ['position', 'difficulty', 'accuracy']
    target_col = 'wpm'
    
    # Drop rows with NaN in key columns
    df_clean = df.dropna(subset=feature_cols + [target_col])
    
    if len(df_clean) < 5:
        print(f"Not enough training examples ({len(df_clean)} < 5). Skipping training.")
        return None
    
    X = df_clean[feature_cols].astype(float)
    y = df_clean[target_col].astype(float)
    
    print(f"Training on {len(X)} examples with features: {feature_cols}")
    print(f"Target: {target_col}")
    
    # Train a simple LightGBM model
    model = lgb.LGBMRegressor(
        num_leaves=31,
        n_estimators=100,
        learning_rate=0.05,
        verbose=-1,
    )
    model.fit(X, y)
    
    # Save model
    with open(output_model, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Trained ranker model saved to: {output_model}")
    
    # Evaluate on training set (quick sanity check)
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Training MSE: {mse:.4f}, R²: {r2:.4f}")
    
    return model


def train_embedding(df: pd.DataFrame, output_model: str = None):
    """Placeholder for fine-tuned embedding model (future)."""
    print("Embedding model training not yet implemented.")
    print("For now, using pre-trained sentence-transformers (all-MiniLM-L6-v2)")
    return None


if __name__ == '__main__':
    model_type = 'ranker'
    output_model = None
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--model-type' and i + 1 < len(sys.argv) - 1:
            model_type = sys.argv[i + 2]
        elif arg == '--output-model' and i + 1 < len(sys.argv) - 1:
            output_model = sys.argv[i + 2]
    
    print(f"Loading training data...")
    df = load_training_data()
    
    if df is None or len(df) == 0:
        print("No training data available.")
        sys.exit(1)
    
    if model_type == 'ranker':
        model = train_ranker(df, output_model)
    elif model_type == 'embedding':
        model = train_embedding(df, output_model)
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)
