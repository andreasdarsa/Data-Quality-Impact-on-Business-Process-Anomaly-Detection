import copy
import random
from datetime import datetime, timedelta


def inject_timeliness_issues(
    logs: list[dict],
    level: float,
    variant: str = "event_delay",
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

        if variant in {"event_delay", "default"}:

            modified = _inject_event_delay(
                case,
                level,
                rng
            )

        elif variant == "clock_drift":

            modified = _inject_clock_drift(
                case,
                level
            )

        else:

            raise ValueError(
                f"Unknown timeliness variant: {variant}"
            )

        if modified:

            _recompute_case_duration(case)

            case["quality_issue"] = "timeliness"
            case["quality_variant"] = variant

    return corrupted


def _parse_timestamp(value: str | None) -> datetime | None:

    if value is None:
        return None

    try:
        return datetime.fromisoformat(value)

    except (TypeError, ValueError):
        return None


def _inject_event_delay(
    case: dict,
    level: float,
    rng: random.Random
) -> bool:

    events = case["events"]

    max_delay_minutes = max(
        1,
        int(level * 60)
    )

    # Identify events with valid completion timestamps.
    valid_indices = []

    for i, event in enumerate(events):

        completion = _parse_timestamp(
            event.get("timestamp_end")
        )

        if completion is not None:
            valid_indices.append(i)

    eligible_events = []

    for position, event_index in enumerate(valid_indices):

        current_timestamp = _parse_timestamp(
            events[event_index].get("timestamp_end")
        )

        # The final event has no subsequent event to constrain its delay.
        if position == len(valid_indices) - 1:

            eligible_events.append(
                (event_index, max_delay_minutes)
            )

            continue

        next_event_index = valid_indices[position + 1]

        next_timestamp = _parse_timestamp(
            events[next_event_index].get("timestamp_end")
        )

        available_gap = (
            next_timestamp - current_timestamp
        ).total_seconds() / 60

        allowed_delay = min(
            max_delay_minutes,
            int(available_gap)
        )

        if allowed_delay >= 1:

            eligible_events.append(
                (event_index, allowed_delay)
            )

    if not eligible_events:
        return False

    event_index, allowed_delay = rng.choice(
        eligible_events
    )

    delay_minutes = rng.randint(
        1,
        allowed_delay
    )

    delay = timedelta(
        minutes=delay_minutes
    )

    _shift_event_timestamps(
        events[event_index],
        delay
    )

    return True


def _inject_clock_drift(
    case: dict,
    level: float
) -> bool:

    drift_step_minutes = max(
        1,
        int(level * 10)
    )

    modified = False

    # Later events receive equal or greater positive shifts.
    # This preserves their chronological order.
    for i, event in enumerate(case["events"]):

        drift = timedelta(
            minutes=i * drift_step_minutes
        )

        if _shift_event_timestamps(event, drift):
            modified = True

    return modified


def _shift_event_timestamps(
    event: dict,
    shift: timedelta
) -> bool:

    modified = False

    for field in ("timestamp_start", "timestamp_end"):

        timestamp = _parse_timestamp(
            event.get(field)
        )

        if timestamp is None:
            continue

        event[field] = (
            timestamp + shift
        ).isoformat()

        modified = True

    return modified


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
