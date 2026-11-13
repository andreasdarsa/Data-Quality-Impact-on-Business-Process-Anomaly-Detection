from datetime import datetime

import numpy as np


def extract_temporal_features(case: dict) -> dict:
    events = case["events"]

    # Durations are expressed in seconds.
    # Missing durations are excluded.
    durations = [
        float(event["duration"])
        for event in events
        if event.get("duration") is not None
        and event["duration"] >= 0
    ]

    # Calculate inter-event gaps.
    # BPIC12: previous completion -> current start.
    # BPIC13: previous timestamp -> current timestamp.
    gaps = []

    for prev, curr in zip(events, events[1:]):
        prev_timestamp = prev.get("timestamp_end")

        curr_timestamp = (
            curr.get("timestamp_start")
            or curr.get("timestamp_end")
        )

        if not prev_timestamp or not curr_timestamp:
            continue

        try:
            prev_dt = datetime.fromisoformat(prev_timestamp)
            curr_dt = datetime.fromisoformat(curr_timestamp)
        except (TypeError, ValueError):
            continue

        gap_minutes = (
            curr_dt - prev_dt
        ).total_seconds() / 60

        # Preserve negative gaps as potential anomaly signals.
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
