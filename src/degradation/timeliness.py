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
    selected_indices = rng.sample(range(len(corrupted)), n_corrupted)

    for idx in selected_indices:
        case = corrupted[idx]

        if variant in {"event_delay", "default"}:
            _inject_event_delay(case, level, rng)

        elif variant == "event_shuffle":
            _shuffle_events(case, rng)

        elif variant == "clock_drift":
            _inject_clock_drift(case, level, rng)

        else:
            raise ValueError(f"Unknown timeliness variant: {variant}")

        case["quality_issue"] = "timeliness"
        case["quality_variant"] = variant

    return corrupted


def _inject_event_delay(case: dict, level: float, rng: random.Random) -> None:
    if not case["events"]:
        return

    event = rng.choice(case["events"])
    delay_minutes = max(1, int(level * 60))

    start = datetime.fromisoformat(event["timestamp_start"])
    end = datetime.fromisoformat(event["timestamp_end"])

    event["timestamp_start"] = (start + timedelta(minutes=delay_minutes)).isoformat()
    event["timestamp_end"] = (end + timedelta(minutes=delay_minutes)).isoformat()


def _shuffle_events(case: dict, rng: random.Random) -> None:
    if len(case["events"]) > 2:
        rng.shuffle(case["events"])


def _inject_clock_drift(case: dict, level: float, rng: random.Random) -> None:
    drift_step = max(1, int(level * 10))

    for i, event in enumerate(case["events"]):
        drift = timedelta(minutes=i * drift_step)

        start = datetime.fromisoformat(event["timestamp_start"])
        end = datetime.fromisoformat(event["timestamp_end"])

        event["timestamp_start"] = (start + drift).isoformat()
        event["timestamp_end"] = (end + drift).isoformat()
