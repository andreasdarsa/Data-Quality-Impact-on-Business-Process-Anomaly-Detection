import copy
import random
import math
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


def _parse_timestamp(value: str | None) -> datetime | None:

    if value is None:
        return None

    try:
        return datetime.fromisoformat(value)

    except (TypeError, ValueError):
        return None


def _shift_timestamps(
    case: dict,
    level: float,
    rng: random.Random
) -> None:

    max_shift_minutes = max(1, int(level * 30))

    events = case["events"]

    completion_times = [
        _parse_timestamp(event.get("timestamp_end"))
        for event in events
    ]

    valid_indices = [
        i for i, timestamp in enumerate(completion_times)
        if timestamp is not None
    ]

    previous_shifted_timestamp = None

    for position, event_index in enumerate(valid_indices):

        current_timestamp = completion_times[event_index]

        min_shift = -max_shift_minutes
        max_shift = max_shift_minutes

        # Ensure that the current event does not precede
        # the previously shifted event.
        if previous_shifted_timestamp is not None:

            difference = (
                previous_shifted_timestamp - current_timestamp
            ).total_seconds() / 60

            min_shift = max(
                min_shift,
                math.ceil(difference)
            )

        # Reserve enough space for the next event to receive
        # its maximum possible negative shift.
        if position < len(valid_indices) - 1:

            next_timestamp = completion_times[
                valid_indices[position + 1]
            ]

            difference = (
                next_timestamp - current_timestamp
            ).total_seconds() / 60

            max_shift = min(
                max_shift,
                math.floor(difference) - max_shift_minutes
            )

        if min_shift > max_shift:
            shift_minutes = 0
        else:
            shift_minutes = rng.randint(
                min_shift,
                max_shift
            )

        shift = timedelta(minutes=shift_minutes)

        _shift_event_timestamps(
            events[event_index],
            shift
        )

        previous_shifted_timestamp = (
            current_timestamp + shift
        )


def _shift_event_timestamps(
    event: dict,
    shift: timedelta
) -> None:

    for field in ("timestamp_start", "timestamp_end"):

        timestamp = _parse_timestamp(
            event.get(field)
        )

        if timestamp is None:
            continue

        event[field] = (
            timestamp + shift
        ).isoformat()


def _recompute_case_duration(case: dict) -> None:

    completion_timestamps = []

    for event in case["events"]:

        timestamp = _parse_timestamp(
            event.get("timestamp_end")
        )

        if timestamp is not None:
            completion_timestamps.append(timestamp)

    if len(completion_timestamps) >= 2:

        case["total_duration"] = (
            max(completion_timestamps)
            - min(completion_timestamps)
        ).total_seconds()

    elif len(completion_timestamps) == 1:

        case["total_duration"] = 0.0
