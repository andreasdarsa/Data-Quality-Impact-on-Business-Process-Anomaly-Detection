import random
from datetime import datetime, timedelta


def inject_anomaly(case, anomaly_group):

    if anomaly_group == "structural":
        return random.choice([
            structural_wrong_order,
            structural_missing_start,
            structural_double_end,
            structural_missing_and_wrong
        ])(case)

    if anomaly_group == "temporal":
        return random.choice([
            temporal_long_duration,
            temporal_short_duration,
            temporal_negative,
            temporal_gap
        ])(case)

    if anomaly_group == "contextual":
        return random.choice([
            contextual_wrong_resource,
            contextual_out_of_range,
            contextual_priority_duration
        ])(case)

    return case


def recompute_case_times(case):
    events = case["events"]

    for i in range(len(events)):
        start = datetime.fromisoformat(events[i]["timestamp_start"])
        duration = events[i]["duration"]

        end = start + timedelta(minutes=duration)
        events[i]["timestamp_end"] = end.isoformat()

        if i < len(events) - 1:
            next_start = datetime.fromisoformat(events[i+1]["timestamp_start"])
            gap = next_start - datetime.fromisoformat(events[i]["timestamp_end"])
            events[i+1]["timestamp_start"] = (end + gap).isoformat()

    case["total_duration"] = sum(e["duration"] for e in events)
    return case


# -------------------
# Structural
# -------------------

def structural_wrong_order(case):
    events = case["events"]

    if len(events) > 3:
        i, j = sorted(random.sample(range(len(events)), 2))
        events[i], events[j] = events[j], events[i]

    case = recompute_case_times(case)
    case["is_anomaly"] = True
    case["subtype"] = "structural_wrong_order"
    return case


def structural_missing_start(case):
    if case["events"]:
        case["events"].pop(0)

    case = recompute_case_times(case)
    case["is_anomaly"] = True
    case["subtype"] = "structural_missing_start"
    return case


def structural_double_end(case):
    last = case["events"][-1].copy()
    last["duration"] *= random.randint(1, 3)

    case["events"].append(last)

    case = recompute_case_times(case)
    case["is_anomaly"] = True
    case["subtype"] = "structural_double_end"
    return case


def structural_missing_and_wrong(case):
    events = case["events"]

    if len(events) > 3:
        events.pop(random.randint(1, len(events)-2))

        i, j = sorted(random.sample(range(len(events)), 2))
        events[i], events[j] = events[j], events[i]

    case["events"] = events

    case = recompute_case_times(case)
    case["is_anomaly"] = True
    case["subtype"] = "structural_missing_and_wrong"
    return case


# -------------------
# Temporal
# -------------------

def temporal_long_duration(case):
    e = random.choice(case["events"])
    e["duration"] *= random.randint(3, 10)

    return finalize(case, "temporal_long_duration")


def temporal_short_duration(case):
    e = random.choice(case["events"])
    e["duration"] = random.randint(1, 2)

    return finalize(case, "temporal_short_duration")


def temporal_negative(case):
    e = random.choice(case["events"])
    e["timestamp_start"], e["timestamp_end"] = e["timestamp_end"], e["timestamp_start"]

    case["is_anomaly"] = True
    case["subtype"] = "temporal_negative"
    return case


def temporal_gap(case):
    if len(case["events"]) > 2:
        idx = random.randint(1, len(case["events"]) - 2)

        start = datetime.fromisoformat(case["events"][idx]["timestamp_start"])
        case["events"][idx]["timestamp_start"] = (start + timedelta(minutes=120)).isoformat()

    return finalize(case, "temporal_gap")


# -------------------
# Contextual
# -------------------

def contextual_wrong_resource(case):
    for e in case["events"]:
        if e["activity"] == "D":
            e["resource"] = random.choice(["junior", "system"])

    case["is_anomaly"] = True
    case["subtype"] = "contextual_wrong_resource"
    return case


def contextual_out_of_range(case):
    case["amount"] = random.randint(20000, 80000)

    case["is_anomaly"] = True
    case["subtype"] = "contextual_out_of_range"
    return case


def contextual_priority_duration(case):
    case["priority"] = "high"

    for e in case["events"]:
        e["duration"] *= random.randint(3, 6)

    return finalize(case, "contextual_priority_duration")


# -------------------
# Helper
# -------------------

def finalize(case, subtype):
    case = recompute_case_times(case)
    case["is_anomaly"] = True
    case["subtype"] = subtype
    return case
