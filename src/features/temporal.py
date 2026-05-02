from datetime import datetime
import numpy as np


def extract_temporal_features(case: dict) -> dict:
    events = case["events"]

    durations = [e.get("duration", 0) for e in events]

    gaps = []
    for prev, curr in zip(events, events[1:]):
        prev_end = datetime.fromisoformat(prev["timestamp_end"])
        curr_start = datetime.fromisoformat(curr["timestamp_start"])
        gaps.append((curr_start - prev_end).total_seconds() / 60)

    return {
        "mean_duration": float(np.mean(durations)) if durations else 0,
        "std_duration": float(np.std(durations)) if durations else 0,
        "min_duration": float(np.min(durations)) if durations else 0,
        "max_duration": float(np.max(durations)) if durations else 0,
        "mean_gap": float(np.mean(gaps)) if gaps else 0,
        "std_gap": float(np.std(gaps)) if gaps else 0,
        "min_gap": float(np.min(gaps)) if gaps else 0,
        "max_gap": float(np.max(gaps)) if gaps else 0,
    }
