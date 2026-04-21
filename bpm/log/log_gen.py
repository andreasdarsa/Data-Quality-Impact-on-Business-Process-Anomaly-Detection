import random as rnd
import json
from datetime import datetime, timedelta
from bpm.log.anomalies import inject_anomaly

SEED = 42
rnd.seed(SEED)

ACTIVITY_DURATION = {
    "A": (2, 5),
    "B": (3, 7),
    "C": (5, 15),
    "D": (4, 10),
    "E": (2, 4),
    "G": (3, 8)
}

RESOURCE_MAP = {
    "A": ["junior", "senior"],
    "B": ["junior", "senior"],
    "C": ["junior", "senior"],
    "D": ["senior"],
    "E": ["senior"],
    "G": ["system"]
}

BASE_FLOWS = [
    ["A", "B", "C", "D", "E"],
    ["A", "B", "C", "C", "D", "E"],
    ["A", "B", "D", "E"],
    ["G", "A", "B", "C", "D", "E"]
]

PRIORITY_RULES = {
    "low": {"amount": (100, 1000), "target_duration": (40, 120)},
    "normal": {"amount": (1000, 5000), "target_duration": (20, 60)},
    "high": {"amount": (5000, 10000), "target_duration": (5, 30)}
}

ANOMALIES = [
    ("structural", 400),
    ("temporal", 400),
    ("contextual", 400)
]

# -------------------
# Helpers
# -------------------

def randomize_flow(flow):
    flow = flow.copy()

    # μικρές διαφοροποιήσεις
    if rnd.random() < 0.3:
        flow.insert(rnd.randint(1, len(flow)-1), rnd.choice(["B", "C"]))

    if rnd.random() < 0.2 and "C" in flow:
        flow.remove("C")

    return flow


def create_event(activity, start_time):
    duration = rnd.randint(*ACTIVITY_DURATION[activity])
    duration += rnd.randint(-1, 2)  # noise

    duration = max(1, duration)

    end_time = start_time + timedelta(minutes=duration)

    return {
        "activity": activity,
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration": duration,
        "resource": rnd.choice(RESOURCE_MAP[activity])
    }, end_time


# -------------------
# Case generation
# -------------------

def generate_case(case_id):
    flow = randomize_flow(rnd.choice(BASE_FLOWS))

    events = []

    current_time = datetime(2026, 1, 1, 8, 0, 0) + timedelta(
        minutes=rnd.randint(0, 10000)
    )

    priority = rnd.choices(
        ["low", "normal", "high"],
        weights=[0.4, 0.4, 0.2]
    )[0]

    amount = rnd.randint(*PRIORITY_RULES[priority]["amount"])

    for activity in flow:
        event, end_time = create_event(activity, current_time)
        events.append(event)

        gap = rnd.randint(1, 10)
        current_time = end_time + timedelta(minutes=gap)

    total_duration = sum(e["duration"] for e in events)

    return {
        "case_id": case_id,
        "events": events,
        "amount": amount,
        "priority": priority,
        "total_duration": total_duration,
        "is_anomaly": False,
        "subtype": "normal"
    }


def build_log():
    cases = []
    case_counter = 1

    # Normal cases
    for _ in range(8000):
        cases.append(generate_case(f"C{case_counter:05d}"))
        case_counter += 1

    # Anomalies
    for anomaly_group, count in ANOMALIES:
        for _ in range(count):
            case = generate_case(f"C{case_counter:05d}")

            # πιθανότητα multi anomaly
            num_anomalies = rnd.choice([1, 1, 2])

            for _ in range(num_anomalies):
                case = inject_anomaly(case, anomaly_group)

            cases.append(case)
            case_counter += 1

    rnd.shuffle(cases)

    with open("bpm/data/json/baseline_event_log.json", "w") as f:
        json.dump(cases, f, indent=2)

    print("Dataset created:", len(cases))