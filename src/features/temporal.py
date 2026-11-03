from datetime import datetime

import numpy as np


def extract_temporal_features(case: dict) -> dict:
    events = case["events"]

    # Durations are expressed in seconds.
    durations = [
        float(event["duration"])
        for event in events
        if event.get("duration") is not None
        and event["duration"] > 0
    ]

    # Calculate gaps only when both timestamps are available.
    gaps = []

    for prev, curr in zip(events, events[1:]):
        prev_end = prev.get("timestamp_end")
        curr_start = curr.get("timestamp_start")

        if not prev_end or not curr_start:
            continue

        try:
            prev_end_dt = datetime.fromisoformat(prev_end)
            curr_start_dt = datetime.fromisoformat(curr_start)
        except (TypeError, ValueError):
            continue

        gap_minutes = (
            curr_start_dt - prev_end_dt
        ).total_seconds() / 60

        gaps.append(gap_minutes)

    return {
        "mean_duration": float(np.mean(durations)) if durations else 0.0,
        "std_duration": float(np.std(durations)) if durations else 0.0,
        "min_duration": float(np.min(durations)) if durations else 0.0,
        "max_duration": float(np.max(durations)) if durations else 0.0,
        "mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "std_gap": float(np.std(gaps)) if gaps else 0.0,
        "min_gap": float(np.min(gaps)) if gaps else 0.0,
        "max_gap": float(np.max(gaps)) if gaps else 0.0,
    }