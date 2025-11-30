import numpy as np
import json

def compute_feature_stats(feature_matrix):
    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = feature_matrix[:, i]
        stats[name] = {
            "mean": float(col.mean()),
            "std": float(col.std() + 1e-8)
        }
    with open("feature_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

def normalize_features(features, stats):
    vec = []
    for k in FEATURE_NAMES:
        mean = stats[k]["mean"]
        std = stats[k]["std"]
        vec.append((features[k] - mean) / std)
    return np.array(vec, dtype=np.float32)

x = normalize_features(raw_features, loaded_stats)  # numpy
embedding = item_tower(x)  # torch/tf model

mlp = nn.Sequential(
    nn.Linear(33, 128),
    nn.LayerNorm(128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32)
)
