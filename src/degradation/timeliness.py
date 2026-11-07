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
    eligible_events = [
        event for event in case["events"]
        if (
            _parse_timestamp(event.get("timestamp_start")) is not None
            or _parse_timestamp(event.get("timestamp_end")) is not None
        )
    ]

    if not eligible_events:
        return False

    event = rng.choice(eligible_events)

    # Delay increases with degradation level.
    max_delay_minutes = max(1, int(level * 60))
    delay_minutes = rng.randint(1, max_delay_minutes)

    delay = timedelta(minutes=delay_minutes)

    _shift_event_timestamps(event, delay)

    return True


def _inject_clock_drift(
    case: dict,
    level: float
) -> bool:
    drift_step_minutes = max(1, int(level * 10))
    modified = False

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
        timestamp = _parse_timestamp(event.get(field))

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
