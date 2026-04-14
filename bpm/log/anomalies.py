import random
from datetime import datetime, timedelta

VALID_FLOWS = [
    ["A", "B", "C", "D", "E"],
    ["A", "B", "C", "C", "D", "E"],
    ["A", "B", "D", "E"],
    ["G", "A", "B", "C", "D", "E"]
]


def recompute_case_times(case):

    events = case["events"]

    for i in range(len(events)):

        start = datetime.fromisoformat(events[i]["timestamp_start"])
        duration = events[i]["duration"]

        end = start + timedelta(minutes=duration)

        events[i]["timestamp_end"] = end.isoformat()

        if i < len(events) - 1:
            next_start = datetime.fromisoformat(events[i+1]["timestamp_start"])

            # keep the gap
            gap = next_start - datetime.fromisoformat(events[i]["timestamp_end"])
            new_next_start = end + gap

            events[i+1]["timestamp_start"] = new_next_start.isoformat()

    case["total_duration"] = sum(e["duration"] for e in events)

    return case


"""
Function imported by the log generator in order to inject anomalies into the generated cases.
Each anomaly type has a corresponding function that modifies the case in a specific way to create the anomaly.
"""


def inject_anomaly(case: dict, anomaly_type: str):
    if anomaly_type == "structural_wrong_order":
        return structural_wrong_order(case)

    if anomaly_type == "structural_missing_start":
        return structural_missing_start(case)

    if anomaly_type == "structural_double_end":
        return structural_double_end(case)

    if anomaly_type == "structural_missing_and_wrong":
        return structural_missing_and_wrong(case)

    if anomaly_type == "temporal_long_duration":
        return temporal_long_duration(case)

    if anomaly_type == "temporal_short_duration":
        return temporal_short_duration(case)

    if anomaly_type == "temporal_negative":
        return temporal_negative(case)

    if anomaly_type == "temporal_gap":
        return temporal_gap(case)

    if anomaly_type == "contextual_wrong_resource":
        return contextual_wrong_resource(case)

    if anomaly_type == "contextual_out_of_range":
        return contextual_out_of_range(case)

    if anomaly_type == "contextual_priority_duration":
        return contextual_priority_duration(case)


"""
Activities that appear at the wrong order (e.g., B before A)
"""


def structural_wrong_order(case):

    events = case["events"]

    if len(events) >= 4:
        # C after D
        idx_c = next((i for i,e in enumerate(events) if e["activity"]=="C"), None)
        idx_d = next((i for i,e in enumerate(events) if e["activity"]=="D"), None)

        if idx_c is not None and idx_d is not None and idx_d > idx_c:
            events[idx_c], events[idx_d] = events[idx_d], events[idx_c]

    case = recompute_case_times(case)

    case["is_anomaly"] = True
    case["anomaly_type"] = "structural_wrong_order"

    return case


"""
Missing starting activity (A or G)
"""


def structural_missing_start(case):

    events = case["events"]

    if events and events[0]["activity"] in ["A","G"]:
        events.pop(0)

    case = recompute_case_times(case)

    case["is_anomaly"] = True
    case["anomaly_type"] = "missing_start"

    return case


"""
Last activity (E) appears twice
"""


def structural_double_end(case):

    last_event = case["events"][-1]

    duplicated = last_event.copy()
    duplicated["duration"] *= random.randint(1, 2)

    case["events"].append(duplicated)

    case = recompute_case_times(case)

    case["is_anomaly"] = True
    case["anomaly_type"] = "duplicate_end"

    return case


"""
Activities that are missing and appear in the wrong order
"""


def structural_missing_and_wrong(case):

    events = case["events"]

    if len(events) >= 4:

        # remove C
        events = [e for e in events if e["activity"] != "C"]

        # D before B
        if len(events) >= 3:
            events[1], events[2] = events[2], events[1]

        case["events"] = events

    case = recompute_case_times(case)

    case["is_anomaly"] = True
    case["anomaly_type"] = "missing_and_wrong_order"

    return case


"""
Activities with unusually long duration (e.g., 5-10 times the normal duration)
"""


def temporal_long_duration(case):

    event = random.choice(case["events"])
    event["duration"] *= random.randint(5, 10)

    case = recompute_case_times(case)

    case["is_anomaly"] = True
    case["anomaly_type"] = "long_duration"

    return case


"""
Activities with unusually short duration (e.g., 1)
"""


def temporal_short_duration(case):

    event = random.choice(case["events"])
    event["duration"] = 1

    case = recompute_case_times(case)

    case["is_anomaly"] = True
    case["anomaly_type"] = "short_duration"

    return case


"""
Activities with negative duration (end time before start time)
"""


def temporal_negative(case):

    event = random.choice(case["events"])

    event["timestamp_start"], event["timestamp_end"] = (
        event["timestamp_end"],
        event["timestamp_start"]
    )

    case["is_anomaly"] = True
    case["anomaly_type"] = "negative_duration"

    return case


"""
Activities with an artificial gap (e.g., 8 hours) between them
"""


def temporal_gap(case):

    events = case["events"]

    if len(events) >= 3:

        gap = timedelta(minutes=120)

        start = datetime.fromisoformat(events[2]["timestamp_start"])
        events[2]["timestamp_start"] = (start + gap).isoformat()

    case = recompute_case_times(case)

    case["is_anomaly"] = True
    case["anomaly_type"] = "artificial_gap"

    return case


"""
Activities that are assigned to the wrong resource (e.g., D assigned to junior instead of senior)
"""


def contextual_wrong_resource(case):

    for event in case["events"]:
        if event["activity"] == "D":
            event["resource"] = "junior"

    case["is_anomaly"] = True
    case["anomaly_type"] = "wrong_resource"

    return case


"""
Activities with out-of-range attribute values (e.g., amount = 50000)
"""


def contextual_out_of_range(case):

    case["amount"] = random.randint(20000, 50000)

    case["is_anomaly"] = True
    case["anomaly_type"] = "out_of_range"

    return case


def contextual_priority_duration(case):

    case["priority"] = "high"

    for event in case["events"]:
        event["duration"] *= random.randint(5, 8)

    case = recompute_case_times(case)

    case["is_anomaly"] = True
    case["anomaly_type"] = "priority_duration_conflict"

    return case

