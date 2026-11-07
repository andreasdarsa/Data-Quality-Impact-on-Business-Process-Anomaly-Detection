import copy
import random
from datetime import datetime, timedelta


def inject_accuracy_noise(
    logs: list[dict],
    level: float,
    variant: str = "duration_timestamp",
    seed: int = 42
) -> list[dict]:
    rng = random.Random(seed)
    corrupted = copy.deepcopy(logs)

    n_corrupted = int(len(corrupted) * level)
    selected_indices = rng.sample(
        range(len(corrupted)),
        n_corrupted
    )

    for idx in selected_indices:
        case = corrupted[idx]

        if variant in {"duration", "duration_timestamp", "default"}:
            _corrupt_durations(case, level, rng)

        if variant in {"timestamp", "duration_timestamp", "default"}:
            _shift_timestamps(case, level, rng)

        _recompute_case_duration(case)

        case["quality_issue"] = "accuracy"
        case["quality_variant"] = variant

    return corrupted


def _corrupt_durations(
    case: dict,
    level: float,
    rng: random.Random
) -> None:
    for event in case["events"]:
        duration = event.get("duration")

        if duration is None:
            continue

        factor = 1 + rng.uniform(-level, level)

        event["duration"] = max(
            0,
            round(float(duration) * factor)
        )


def _shift_timestamps(
    case: dict,
    level: float,
    rng: random.Random
) -> None:
    max_shift_minutes = max(1, int(level * 30))

    for event in case["events"]:
        shift = rng.randint(
            -max_shift_minutes,
            max_shift_minutes
        )

        for field in ("timestamp_start", "timestamp_end"):
            timestamp = event.get(field)

            if timestamp is None:
                continue

            try:
                parsed = datetime.fromisoformat(timestamp)
            except (TypeError, ValueError):
                continue

            event[field] = (
                parsed + timedelta(minutes=shift)
            ).isoformat()


def _recompute_case_duration(case: dict) -> None:
    completion_timestamps = []

    for event in case["events"]:
        timestamp = event.get("timestamp_end")

        if timestamp is None:
            continue

        try:
            completion_timestamps.append(
                datetime.fromisoformat(timestamp)
            )
        except (TypeError, ValueError):
            continue

    if len(completion_timestamps) >= 2:
        case["total_duration"] = (
            max(completion_timestamps)
            - min(completion_timestamps)
        ).total_seconds()

    elif len(completion_timestamps) == 1:
        case["total_duration"] = 0.0
