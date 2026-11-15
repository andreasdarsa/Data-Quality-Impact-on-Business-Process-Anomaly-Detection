
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.loader import load_logs
from src.degradation.accuracy import inject_accuracy_noise
from src.degradation.timeliness import inject_timeliness_issues
from src.process_mining.dfg import build_dfg, prune_dfg
from src.features.extractor import extract_features

# Load BPIC13 test split
logs = load_logs(
    Path("data/processed/bpic13_test_seed_42.json")
)

# Build DFG from normal training cases
train_logs = load_logs(
    Path("data/processed/bpic13_train_seed_42.json")
)

normal_train = [
    case for case in train_logs
    if not case.get("is_anomaly", False)
]

dfg = prune_dfg(build_dfg(normal_train), min_freq=2)

# Extract clean features
X_clean, _, _ = extract_features(
    logs, dfg, dataset="bpic13"
)

temporal_features = [
    "total_duration",
    "mean_gap",
    "std_gap",
    "min_gap",
    "max_gap",
]

temporal_features = [
    f for f in temporal_features if f in X_clean.columns
]

print("Clean feature matrix:", X_clean.shape)
print("\nClean temporal feature statistics:")
print(X_clean[temporal_features].describe().round(3))

# Test each degradation dimension and level
results = []

for dimension in ("accuracy", "timeliness"):

    for level in (0.1, 0.2, 0.3):

        if dimension == "accuracy":
            degraded = inject_accuracy_noise(
                logs, level, "duration_timestamp", seed=101
            )
        else:
            degraded = inject_timeliness_issues(
                logs, level, "event_delay", seed=101
            )

        X_degraded, _, _ = extract_features(
            degraded, dfg, dataset="bpic13"
        )

        differences = X_degraded[temporal_features] - X_clean[temporal_features]

        changed_cases = int(
            (differences.abs().max(axis=1) > 1e-9).sum()
        )

        row = {
            "dimension": dimension,
            "level": level,
            "metadata_cases": sum(
                case.get("quality_issue") == dimension
                for case in degraded
            ),
            "changed_feature_cases": changed_cases,
        }

        for feature in temporal_features:
            row[f"{feature}_mean_abs_diff"] = (
                differences[feature].abs().mean()
            )

        results.append(row)

df = pd.DataFrame(results)

print("\nDEGRADATION FEATURE IMPACT")
print(df.round(4).to_string(index=False))
