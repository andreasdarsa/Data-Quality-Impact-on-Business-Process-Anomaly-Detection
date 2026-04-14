import random as rnd
import json
from datetime import datetime, timedelta
from bpm.log.anomalies import inject_anomaly

# -------------------
# Activity definitions
# -------------------

# Minimum and maximum duration (in minutes) for each activity
ACTIVITY_DURATION = {
    "A": (2, 5),
    "B": (3, 7),
    "C": (5, 15),
    "D": (4, 10),
    "E": (2, 4),
    "G": (3, 8)
}

# Allowed resources for each activity
RESOURCE_MAP = {
    "A": ["junior", "senior"],
    "B": ["junior", "senior"],
    "C": ["junior", "senior"],
    "D": ["senior"],
    "E": ["senior"],
    "G": ["system"]
}

# -------------------
# Process flows
# -------------------

DOMINANT_FLOW = ["A","B","C","D","E"]
LOOP_C_FLOW = ["A","B","C","C","D","E"]
SKIP_C_FLOW = ["A","B","D","E"]
G_FLOW = ["G","A","B","C","D","E"]

# -------------------
# Semantic priority rules
# -------------------

PRIORITY_RULES = {
    "low": {
        "amount": (100, 1000),
        "target_duration": (40, 120)
    },
    "normal": {
        "amount": (1000, 5000),
        "target_duration": (20, 60)
    },
    "high": {
        "amount": (5000, 10000),
        "target_duration": (5, 30)
    }
}

ANOMALIES = [
        ("structural_wrong_order",10),
        ("structural_missing_start",10),
        ("structural_double_end",10),
        ("structural_missing_and_wrong",10),

        ("temporal_long_duration",10),
        ("temporal_short_duration",10),
        ("temporal_negative",8),
        ("temporal_gap",8),

        ("contextual_wrong_resource",8),
        ("contextual_out_of_range",8),
        ("contextual_priority_duration",8),
    ]

# -------------------
# Event creation
# -------------------

def create_event(activity, start_time):

    duration = rnd.randint(*ACTIVITY_DURATION[activity])
    end_time = start_time + timedelta(minutes=duration)

    event = {
        "activity": activity,
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration": duration,
        "resource": RESOURCE_MAP[activity][rnd.randint(0, len(RESOURCE_MAP[activity]) - 1)]
    }

    return event, end_time


# -------------------
# Case generation
# -------------------

def generate_case(case_id, flow):

    events = []

    current_time = datetime(2026,1,1,8,0,0) + timedelta(
        minutes=rnd.randint(0, 5000)
    )

    priority = rnd.choices(
        ["low", "normal", "high"],
        weights=[0.4, 0.4, 0.2]
    )[0]

    amount = rnd.randint(*PRIORITY_RULES[priority]["amount"])

    for activity in flow:

        event, end_time = create_event(activity, current_time)

        events.append(event)

        gap = rnd.randint(1,5)
        current_time = end_time + timedelta(minutes=gap)

    case_start = datetime.fromisoformat(events[0]["timestamp_start"])
    case_end = datetime.fromisoformat(events[-1]["timestamp_end"])

    total_duration = (case_end - case_start).total_seconds() / 60

    min_dur, max_dur = PRIORITY_RULES[priority]["target_duration"]

    if total_duration < min_dur or total_duration > max_dur:

        target = rnd.randint(min_dur, max_dur)
        scale_factor = target / total_duration if total_duration > 0 else 1

        current_time = case_start
        new_events = []

        for event in events:

            scaled_duration = max(1, int(event["duration"] * scale_factor))
            end_time = current_time + timedelta(minutes=scaled_duration)

            new_event = event.copy()
            new_event["timestamp_start"] = current_time.isoformat()
            new_event["timestamp_end"] = end_time.isoformat()
            new_event["duration"] = scaled_duration

            new_events.append(new_event)

            gap = rnd.randint(1, 5)
            current_time = end_time + timedelta(minutes=gap)

        events = new_events

    total_duration = sum(e["duration"] for e in events)

    case = {
        "case_id": case_id,
        "events": events,
        "amount": amount,
        "priority": priority,
        "total_duration": total_duration,
        "is_anomaly": False
    }

    return case


def add_cases(n, flow, subtype, case_counter):
    cases = []

    for _ in range(n):

        case = generate_case(f"C{case_counter:04d}", flow)
        case["subtype"] = subtype

        cases.append(case)

        case_counter += 1

    return cases, case_counter


def build_log():
    cases = []
    case_counter = 1

    # Normal distributions
    new_cases, case_counter = add_cases(580, DOMINANT_FLOW, "dominant", case_counter)
    cases.extend(new_cases)
    new_cases, case_counter = add_cases(140, LOOP_C_FLOW, "loop_C", case_counter)
    cases.extend(new_cases)
    new_cases, case_counter = add_cases(90, SKIP_C_FLOW, "skip_C", case_counter)
    cases.extend(new_cases)
    new_cases, case_counter = add_cases(90, G_FLOW, "include_G", case_counter)
    cases.extend(new_cases)

    # Anomalous cases
    for anomaly_type, count in ANOMALIES:
        for _ in range(count):
            case = generate_case(f"C{case_counter:04d}", DOMINANT_FLOW)
            case["subtype"] = anomaly_type
            anomalous_case = inject_anomaly(case, anomaly_type)
            cases.append(anomalous_case)
            case_counter += 1

    rnd.shuffle(cases)

    with open("bpm/data/json/baseline_event_log.json", "w") as f:
        json.dump(cases, f, indent=2)

    print("Dataset created:", len(cases), "cases")
    print("Dataset saved to bpm/data/json/baseline_event_log.json")
