import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Tuple
import random

from app.database import engine
from app.models.db_models import SnippetUsage, User, Snippet, TelemetrySnippetRaw
from app.ml.user_features import UserFeatureExtractor
from app.ml.loss_formulation import TwoTowerTrainingWrapper, compute_hierarchical_loss
from app.ml.user_encoder import get_model as get_user_model
from app.ml.snippet_tower import get_snippet_model
from app.ml.difficulty_features import compute_difficulty_features # Optional re-calc

# -------------------------------------------------------------------
# 1. Dataset Definition
# -------------------------------------------------------------------

class SnippetRankerDataset(Dataset):
    """
    Fetches valid interaction triplets: (User State, Snippet, Outcome).
    """
    def __init__(self, usage_ids: List[str]):
        self.usage_ids = usage_ids
        self.Session = sessionmaker(bind=engine)

    def __len__(self):
        return len(self.usage_ids)

    def __getitem__(self, idx):
        session = self.Session()
        try:
            # Fetch the usage record
            usage = session.query(SnippetUsage).get(self.usage_ids[idx])
            if not usage:
                return None # Handle in collate

            # Fetch related User and Snippet
            user = session.query(User).get(usage.session.user_id) if usage.session else None
            snippet = session.query(Snippet).get(usage.snippet_id)
            
            if not user or not snippet:
                return None

            # --- A. User Features ---
            # 1. Load Extractor state from DB
            features_dict = user.features or {}
            extractor = UserFeatureExtractor.from_dict(features_dict)
            
            # 2. Long-term Vector (40-dim)
            long_term_vec = extractor.compute_user_features()
            
            # 3. Short-term History Sequence (T, 12)
            # We take the last 10 items from history
            # NOTE: In a real production loop, we might want the history *at the time of interaction*
            # But for now, we use the current stored history as an approximation 
            # or we assume `short_term_history` is append-only and we could index back if we logged timestamps.
            # For MVP: Use current history.
            history_list = extractor.short_term_history[-10:]
            if not history_list:
                short_term_seq = np.zeros((1, 12), dtype=np.float32) # Pad at least 1
            else:
                short_term_seq = np.array(history_list, dtype=np.float32)
            
            # 4. Session Stats (5-dim) - from the specific session of this usage
            # Rolling stats *at that time* are not explicitly stored in SnippetUsage, 
            # but we can approximate from the usage data or stored session aggregate.
            # For MVP: Use usage stats as proxies for "current state"
            wpm_norm = min((usage.user_wpm or 0) / 200.0, 1.0)
            acc_norm = usage.user_accuracy or 0.0
            # We don't have exact backspace/hesitation for this specific snippet in SnippetUsage 
            # unless we joined with KeystrokeEvents or Telemetry.
            # Using defaults/zeros for missing signals for now.
            session_stats = np.array([wpm_norm, acc_norm, 0.0, 0.5, 0.0], dtype=np.float32)

            # --- B. Snippet Features ---
            # Use stored processed embedding if available, else raw normalized features
            # Ideally we want raw normalized features (30-dim) because the SnippetTower learns to project them.
            # If `normalized_features` is None, re-compute (slow fallback).
            if snippet.normalized_features:
                snip_feat = np.array(snippet.normalized_features, dtype=np.float32)
            else:
                snip_feat = np.zeros(30, dtype=np.float32) # Fallback

            # --- C. Targets (Ground Truth) ---
            # We need to reconstruct y_flow, y_growth from the outcome data
            # y_flow (0-1): High accuracy + completion
            # y_growth (-1 to 1): WPM relative to user avg? 
            
            # Calculate y_flow
            # Simple heuristic since we don't have full telemetry in SnippetUsage yet
            # (We'd need TelemetrySnippetRaw for full fidelity)
            y_flow = 1.0 if (usage.user_accuracy > 0.9 and usage.user_wpm > 10) else 0.0
            if usage.user_accuracy < 0.8: y_flow = 0.0
            
            # Calculate y_growth
            # Compare usage WPM to user's overall WPM (from extractor)
            # WPM from extractor might be slightly "future" relative to this sample, but acceptable for MVP
            user_avg_wpm = extractor.wpm_history[-1] if extractor.wpm_history else 40.0
            # Avoid div by zero
            baseline = max(10.0, user_avg_wpm)
            wpm_delta = (usage.user_wpm - baseline) / baseline
            y_growth = np.clip(wpm_delta, -1.0, 1.0)

            # Difficulty Boundaries
            comfort, struggle = extractor.get_difficulty_boundaries()
            # Snippet difficulty
            d_snippet = (snippet.difficulty_score or 5.0) / 10.0
            
            return {
                "user_long_term": long_term_vec,      # (40,)
                "user_short_term": short_term_seq,    # (T, 12)
                "user_session": session_stats,        # (5,)
                "snippet_features": snip_feat,        # (30,)
                "targets": {
                    "y_flow": float(y_flow),
                    "y_growth": float(y_growth),
                    "d_snippet": float(d_snippet),
                    "c_user": float(comfort),
                    "s_user": float(struggle)
                }
            }

        finally:
            session.close()

# -------------------------------------------------------------------
# 2. Collate Function (Batching)
# -------------------------------------------------------------------

def collate_ranker_batch(batch):
    """
    Pads sequences and stacks tensors.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # 1. User Long Term & Session
    u_long = torch.stack([torch.from_numpy(b["user_long_term"]) for b in batch])
    u_sess = torch.stack([torch.from_numpy(b["user_session"]) for b in batch])
    
    # 2. User Short Term Sequence (Padding)
    # Find max seq len in this batch
    max_len = max([b["user_short_term"].shape[0] for b in batch])
    padded_seqs = []
    for b in batch:
        seq = b["user_short_term"]
        T, D = seq.shape
        pad_len = max_len - T
        if pad_len > 0:
            # Pre-pad with zeros? Or Post-pad? GRU usually handles either if packed, 
            # but simple zero-padding works. Let's pre-pad (history).
            padding = np.zeros((pad_len, D), dtype=np.float32)
            seq = np.vstack([padding, seq])
        padded_seqs.append(torch.from_numpy(seq))
    
    u_short = torch.stack(padded_seqs) # (B, MaxT, 12)

    # 3. Snippet Features (Positive)
    s_pos = torch.stack([torch.from_numpy(b["snippet_features"]) for b in batch])
    
    # 4. Negatives Generation (In-Batch Negatives + Random)
    # For simplicity, we use In-Batch Negatives: other snippets in the batch are negatives
    # But we need explicit negative tensors for the architecture
    # Let's pick K=3 random negatives per sample from the OTHER snippets in the batch
    # If batch size is small, we might duplicate.
    B = len(batch)
    K = min(3, B-1)
    s_neg_list = []
    
    if K > 0:
        for i in range(B):
            # Indices other than i
            others = [j for j in range(B) if j != i]
            # Sample K
            neg_indices = random.choices(others, k=K)
            negs = s_pos[neg_indices] # (K, 30)
            s_neg_list.append(negs)
        s_neg = torch.stack(s_neg_list) # (B, K, 30)
    else:
        # Fallback if batch=1, just duplicate pos as neg (loss will be 0 for ranking but heads will train)
        # Or generate random noise
        s_neg = torch.randn(B, 1, 30)
    
    # 5. Targets
    y_flow = torch.tensor([[b["targets"]["y_flow"]] for b in batch], dtype=torch.float32)
    y_growth = torch.tensor([[b["targets"]["y_growth"]] for b in batch], dtype=torch.float32)
    d_snippet = torch.tensor([[b["targets"]["d_snippet"]] for b in batch], dtype=torch.float32)
    c_user = torch.tensor([[b["targets"]["c_user"]] for b in batch], dtype=torch.float32)
    s_user = torch.tensor([[b["targets"]["s_user"]] for b in batch], dtype=torch.float32)
    
    return {
        "u_short": u_short,
        "u_sess": u_sess,
        "u_long": u_long,
        "s_pos": s_pos,
        "s_neg": s_neg,
        "targets": {
            "y_flow": y_flow,
            "y_growth": y_growth,
            "d_snippet": d_snippet,
            "c_user": c_user,
            "s_user": s_user
        }
    }


# -------------------------------------------------------------------
# 3. Training Loop
# -------------------------------------------------------------------

def train_ranker(epochs=10, batch_size=32, lr=1e-3):
    print("Initializing Training...")
    
    # 1. Fetch Data IDs
    session = sessionmaker(bind=engine)()
    usage_ids = [u.id for u in session.query(SnippetUsage).all()]
    session.close()
    
    print(f"Found {len(usage_ids)} usage samples.")
    if not usage_ids:
        print("No data found. Exiting.")
        return

    # 2. Dataset & Loader
    dataset = SnippetRankerDataset(usage_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_ranker_batch)
    
    # 3. Model
    user_tower = get_user_model()
    snippet_tower = get_snippet_model()
    model = TwoTowerTrainingWrapper(user_tower, snippet_tower)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        
        for batch in dataloader:
            if not batch: continue
            
            # Forward
            outputs = model(
                batch["u_short"], 
                batch["u_sess"], 
                batch["u_long"], 
                batch["s_pos"], 
                batch["s_neg"]
            )
            
            # Loss
            loss_dict = compute_hierarchical_loss(outputs, batch["targets"], {})
            loss = loss_dict["total"]
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
        
        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # 5. Save
    # For now, we assume the models modify the global singletons in memory or we save them explicitly
    # torch.save(user_tower.state_dict(), "user_tower.pth")
    # torch.save(snippet_tower.state_dict(), "snippet_tower.pth")
    print("Training Complete.")

if __name__ == "__main__":
    train_ranker()
