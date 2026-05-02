import random
from datetime import datetime, timedelta
from typing import Any

from src.generation.anomalies import inject_anomaly


ACTIVITY_DURATION = {
    "A": (2, 5),
    "B": (3, 7),
    "C": (5, 15),
    "D": (4, 10),
    "E": (2, 4),
    "G": (3, 8),
}

RESOURCE_MAP = {
    "A": ["junior", "senior"],
    "B": ["junior", "senior"],
    "C": ["junior", "senior"],
    "D": ["senior"],
    "E": ["senior"],
    "G": ["system"],
}

BASE_FLOWS = [
    ["A", "B", "C", "D", "E"],
    ["A", "B", "C", "C", "D", "E"],
    ["A", "B", "D", "E"],
    ["G", "A", "B", "C", "D", "E"],
]

PRIORITY_RULES = {
    "low": {
        "amount": (100, 1000),
        "target_duration": (40, 120),
    },
    "normal": {
        "amount": (1000, 5000),
        "target_duration": (20, 60),
    },
    "high": {
        "amount": (5000, 10000),
        "target_duration": (5, 30),
    },
}

DEFAULT_ANOMALY_COUNTS = {
    "structural": 400,
    "temporal": 400,
    "contextual": 400,
}


def randomize_flow(flow: list[str], rng: random.Random) -> list[str]:
    flow = flow.copy()

    if rng.random() < 0.3:
        insert_pos = rng.randint(1, len(flow) - 1)
        flow.insert(insert_pos, rng.choice(["B", "C"]))

    if rng.random() < 0.2 and "C" in flow:
        flow.remove("C")

    return flow


def create_event(
    activity: str,
    start_time: datetime,
    rng: random.Random,
) -> tuple[dict[str, Any], datetime]:
    min_duration, max_duration = ACTIVITY_DURATION[activity]

    duration = rng.randint(min_duration, max_duration)
    duration += rng.randint(-1, 2)
    duration = max(1, duration)

    end_time = start_time + timedelta(minutes=duration)

    event = {
        "activity": activity,
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration": duration,
        "resource": rng.choice(RESOURCE_MAP[activity]),
    }

    return event, end_time


def generate_case(
    case_id: str,
    rng: random.Random,
    base_start: datetime | None = None,
) -> dict[str, Any]:
    flow = randomize_flow(rng.choice(BASE_FLOWS), rng)

    if base_start is None:
        base_start = datetime(2026, 1, 1, 8, 0, 0)

    current_time = base_start + timedelta(
        minutes=rng.randint(0, 10000)
    )

    priority = rng.choices(
        ["low", "normal", "high"],
        weights=[0.4, 0.4, 0.2],
        k=1,
    )[0]

    amount = rng.randint(*PRIORITY_RULES[priority]["amount"])

    events = []

    for activity in flow:
        event, end_time = create_event(activity, current_time, rng)
        events.append(event)

        gap = rng.randint(1, 10)
        current_time = end_time + timedelta(minutes=gap)

    total_duration = sum(event["duration"] for event in events)

    return {
        "case_id": case_id,
        "events": events,
        "amount": amount,
        "priority": priority,
        "total_duration": total_duration,
        "is_anomaly": False,
        "subtype": "normal",
    }


def generate_log(
    n_normal: int = 8000,
    anomaly_counts: dict[str, int] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    if anomaly_counts is None:
        anomaly_counts = DEFAULT_ANOMALY_COUNTS

    cases = []
    case_counter = 1

    for _ in range(n_normal):
        case = generate_case(
            case_id=f"C{case_counter:05d}",
            rng=rng,
        )
        cases.append(case)
        case_counter += 1

    for anomaly_group, count in anomaly_counts.items():
        for _ in range(count):
            case = generate_case(
                case_id=f"C{case_counter:05d}",
                rng=rng,
            )

            num_anomalies = rng.choice([1, 1, 2])

            for _ in range(num_anomalies):
                case = inject_anomaly(
                    case=case,
                    anomaly_group=anomaly_group,
                    rng=rng,
                )

            cases.append(case)
            case_counter += 1

    rng.shuffle(cases)

    return cases
