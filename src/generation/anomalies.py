import random
from datetime import datetime, timedelta
from typing import Callable, Any


def inject_anomaly(
    case: dict[str, Any],
    anomaly_group: str,
    rng: random.Random,
) -> dict[str, Any]:
    anomaly_map: dict[str, list[Callable]] = {
        "structural": [
            structural_wrong_order,
            structural_missing_start,
            structural_double_end,
            structural_missing_and_wrong,
        ],
        "temporal": [
            temporal_long_duration,
            temporal_short_duration,
            temporal_negative,
            temporal_gap,
        ],
        "contextual": [
            contextual_wrong_resource,
            contextual_out_of_range,
            contextual_priority_duration,
        ],
    }

    if anomaly_group not in anomaly_map:
        raise ValueError(f"Unknown anomaly group: {anomaly_group}")

    anomaly_fn = rng.choice(anomaly_map[anomaly_group])
    return anomaly_fn(case, rng)


def recompute_case_times(case: dict[str, Any]) -> dict[str, Any]:
    events = case["events"]

    for i, event in enumerate(events):
        start = datetime.fromisoformat(event["timestamp_start"])
        duration = event["duration"]

        end = start + timedelta(minutes=duration)
        event["timestamp_end"] = end.isoformat()

        if i < len(events) - 1:
            next_event = events[i + 1]
            next_start = datetime.fromisoformat(next_event["timestamp_start"])

            old_gap = next_start - datetime.fromisoformat(event["timestamp_end"])
            next_event["timestamp_start"] = (end + old_gap).isoformat()

    case["total_duration"] = sum(event["duration"] for event in events)

    return case


def finalize_anomaly(
    case: dict[str, Any],
    subtype: str,
    recompute_times: bool = True,
) -> dict[str, Any]:
    if recompute_times:
        case = recompute_case_times(case)

    case["is_anomaly"] = True
    case["subtype"] = subtype

    return case


# -------------------
# Structural anomalies
# -------------------

def structural_wrong_order(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    events = case["events"]

    if len(events) > 3:
        i, j = sorted(rng.sample(range(len(events)), 2))
        events[i], events[j] = events[j], events[i]

    return finalize_anomaly(case, "structural_wrong_order")


def structural_missing_start(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    if case["events"]:
        case["events"].pop(0)

    return finalize_anomaly(case, "structural_missing_start")


def structural_double_end(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    if not case["events"]:
        return finalize_anomaly(case, "structural_double_end")

    last = case["events"][-1].copy()
    last["duration"] *= rng.randint(1, 3)

    case["events"].append(last)

    return finalize_anomaly(case, "structural_double_end")


def structural_missing_and_wrong(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    events = case["events"]

    if len(events) > 3:
        remove_idx = rng.randint(1, len(events) - 2)
        events.pop(remove_idx)

        i, j = sorted(rng.sample(range(len(events)), 2))
        events[i], events[j] = events[j], events[i]

    case["events"] = events

    return finalize_anomaly(case, "structural_missing_and_wrong")


# -------------------
# Temporal anomalies
# -------------------

def temporal_long_duration(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    event = rng.choice(case["events"])
    event["duration"] *= rng.randint(3, 10)

    return finalize_anomaly(case, "temporal_long_duration")


def temporal_short_duration(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    event = rng.choice(case["events"])
    event["duration"] = rng.randint(1, 2)

    return finalize_anomaly(case, "temporal_short_duration")


def temporal_negative(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    event = rng.choice(case["events"])

    event["timestamp_start"], event["timestamp_end"] = (
        event["timestamp_end"],
        event["timestamp_start"],
    )

    return finalize_anomaly(
        case,
        "temporal_negative",
        recompute_times=False,
    )


def temporal_gap(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    if len(case["events"]) > 2:
        idx = rng.randint(1, len(case["events"]) - 2)

        start = datetime.fromisoformat(
            case["events"][idx]["timestamp_start"]
        )

        case["events"][idx]["timestamp_start"] = (
            start + timedelta(minutes=120)
        ).isoformat()

    return finalize_anomaly(case, "temporal_gap")


# -------------------
# Contextual anomalies
# -------------------

def contextual_wrong_resource(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    for event in case["events"]:
        if event["activity"] == "D":
            event["resource"] = rng.choice(["junior", "system"])

    return finalize_anomaly(
        case,
        "contextual_wrong_resource",
        recompute_times=False,
    )


def contextual_out_of_range(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    case["amount"] = rng.randint(20000, 80000)

    return finalize_anomaly(
        case,
        "contextual_out_of_range",
        recompute_times=False,
    )


def contextual_priority_duration(
    case: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    case["priority"] = "high"

    for event in case["events"]:
        event["duration"] *= rng.randint(3, 6)

    return finalize_anomaly(case, "contextual_priority_duration")