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
    selected_indices = rng.sample(range(len(corrupted)), n_corrupted)

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


def _corrupt_durations(case: dict, level: float, rng: random.Random) -> None:
    for event in case["events"]:
        factor = 1 + rng.uniform(-level, level)
        event["duration"] = max(1, round(event["duration"] * factor))


def _shift_timestamps(case: dict, level: float, rng: random.Random) -> None:
    max_shift_minutes = max(1, int(level * 30))

    for event in case["events"]:
        shift = rng.randint(-max_shift_minutes, max_shift_minutes)

        start = datetime.fromisoformat(event["timestamp_start"])
        end = datetime.fromisoformat(event["timestamp_end"])

        event["timestamp_start"] = (start + timedelta(minutes=shift)).isoformat()
        event["timestamp_end"] = (end + timedelta(minutes=shift)).isoformat()


def _recompute_case_duration(case: dict) -> None:
    case["total_duration"] = sum(event["duration"] for event in case["events"])
