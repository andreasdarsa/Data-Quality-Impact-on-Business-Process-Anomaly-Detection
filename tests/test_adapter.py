from src.features.extractor import extract_features
from src.data.adapters.binet_adapter import load_and_convert_binet_log, load_binet_log
from src.data.schema import validate_logs
from collections import Counter, defaultdict
from src.features.temporal import extract_temporal_features
from datetime import datetime
from src.features.resource import extract_resource_features
from src.features.control_flow import extract_control_flow_features
from src.process_mining.dfg import build_dfg, prune_dfg

logs = load_and_convert_binet_log("data/bpic12-0.3-1.json")
validate_logs(logs)

print(f"Cases: {len(logs)}")
print(f"Anomalies: {sum(case['is_anomaly'] for case in logs)}")

subtypes = Counter(case["subtype"] for case in logs)

total_events = sum(len(case["events"]) for case in logs)

events_with_start = sum(
    1
    for case in logs
    for event in case["events"]
    if event["timestamp_start"] is not None
)

events_with_end = sum(
    1
    for case in logs
    for event in case["events"]
    if event["timestamp_end"] is not None
)

zero_duration_events = sum(
    1
    for case in logs
    for event in case["events"]
    if event["duration"] == 0
)

print("\n--- Label distribution ---")
for subtype, count in subtypes.most_common():
    print(f"{subtype}: {count}")

print("\n--- Temporal data quality ---")
print(f"Total events: {total_events}")
print(f"Events with start timestamp: {events_with_start}")
print(f"Events with end timestamp: {events_with_end}")
print(f"Events with zero duration: {zero_duration_events}")

raw_data = load_binet_log("data/bpic12-0.3-1.json")

transition_counts = Counter()
examples = {}

for case in raw_data["cases"]:
    for event in case.get("events", []):
        attributes = event.get("attributes", {})
        transition = attributes.get("lifecycle:transition", "<missing>")

        transition_counts[transition] += 1

        if transition not in examples:
            examples[transition] = {
                "activity": attributes.get("concept:name"),
                "timestamp": event.get("timestamp"),
                "timestamp_end": event.get("timestamp_end"),
            }

print("\n--- Lifecycle transitions ---")
for transition, count in transition_counts.most_common():
    print(f"{transition}: {count}")
    print(f"  Example: {examples[transition]}")

transition_pairs = Counter()
examples = {}

for case in raw_data["cases"]:
    activity_transitions = defaultdict(Counter)

    for event in case.get("events", []):
        attrs = event.get("attributes", {})
        activity = attrs.get("concept:name", "<missing>")
        transition = attrs.get("lifecycle:transition", "<missing>")

        activity_transitions[activity][transition] += 1

    for activity, counts in activity_transitions.items():
        key = (
            counts.get("START", 0),
            counts.get("COMPLETE", 0),
            counts.get("SCHEDULE", 0),
        )
        transition_pairs[key] += 1

        if key not in examples:
            examples[key] = {
                "case_id": case.get("id"),
                "activity": activity,
                "counts": dict(counts),
            }

print("\n--- START / COMPLETE / SCHEDULE patterns ---")

for pattern, count in transition_pairs.most_common(15):
    print(f"{pattern}: {count}")
    print(f"  Example: {examples[pattern]}")

# Inspect one case with repeated executions of the same activity.
target_case_id = "173691"

case = next(
    case for case in raw_data["cases"]
    if str(case.get("id")) == target_case_id
)

print(f"\n--- Events for case {target_case_id} ---")

for event in case["events"]:
    attributes = event.get("attributes", {})

    print({
        "activity": attributes.get("concept:name"),
        "transition": attributes.get("lifecycle:transition"),
        "timestamp": event.get("timestamp"),
    })

converted_logs = load_and_convert_binet_log("data/bpic12-0.3-1.json")

dfg = build_dfg(converted_logs)
dfg = prune_dfg(dfg, min_freq=2)

converted_events = [
    event
    for case in converted_logs
    for event in case["events"]
]

print("\n--- Activity-level validation ---")

print(f"Converted events: {len(converted_events)}")

print(
    "Events with missing activity:",
    sum(not event["activity"] for event in converted_events)
)

print(
    "Events with missing end timestamp:",
    sum(event["timestamp_end"] is None for event in converted_events)
)

print(
    "Events with missing start timestamp:",
    sum(event["timestamp_start"] is None for event in converted_events)
)

print(
    "Events with unknown resource:",
    sum(
        event["resource"] in (None, "", "unknown")
        for event in converted_events
    )
)

chronological_cases = 0

for case in converted_logs:
    timestamps = [
        datetime.fromisoformat(event["timestamp_end"])
        for event in case["events"]
        if event["timestamp_end"] is not None
    ]

    if timestamps == sorted(timestamps):
        chronological_cases += 1

print(
    "Chronologically ordered cases:",
    f"{chronological_cases}/{len(converted_logs)}"
)

assert all(event["activity"] for event in converted_events)
assert chronological_cases == len(converted_logs)

print("Activity-level validation PASSED.")

total_events = 0
events_with_end = 0
positive_durations = 0
zero_durations = 0
missing_durations = 0
negative_durations = 0

for case in converted_logs:
    for event in case["events"]:
        total_events += 1

        if event["timestamp_end"] is not None:
            events_with_end += 1

        duration = event["duration"]

        if duration is None:
            missing_durations += 1
        elif duration > 0:
            positive_durations += 1
        elif duration == 0:
            zero_durations += 1
        else:
            negative_durations += 1

print("\n--- Converted temporal data quality ---")
print(f"Total events: {total_events}")
print(f"Events with end timestamp: {events_with_end}")
print(f"Events with positive duration: {positive_durations}")
print(f"Events with zero duration: {zero_durations}")
print(f"Events with missing duration: {missing_durations}")
print(f"Events with negative duration: {negative_durations}")

# Test temporal feature extraction on converted logs
print("\n--- Temporal feature extraction test ---")

test_cases = converted_logs[:100]
successful = 0
failed = 0

for case in test_cases:
    try:
        features = extract_temporal_features(case)

        assert all(
            isinstance(value, (int, float))
            for value in features.values()
        ), f"Non-numeric feature found in case {case['case_id']}"

        successful += 1

    except Exception as exc:
        failed += 1
        print(f"FAILED case {case['case_id']}: {exc}")

print(f"Cases tested: {len(test_cases)}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")

if failed == 0:
    print("Temporal feature extraction test PASSED.")
else:
    print("Temporal feature extraction test FAILED.")

print("\n--- Resource feature extraction test ---")

successful = 0
failed = 0

for case in converted_logs[:100]:
    try:
        features = extract_resource_features(case)

        assert all(
            isinstance(value, (int, float))
            for value in features.values()
        )

        successful += 1

    except Exception as exc:
        failed += 1
        print(f"FAILED case {case['case_id']}: {exc}")

print(f"Cases tested: 100")
print(f"Successful: {successful}")
print(f"Failed: {failed}")

print("\n--- Control-flow feature extraction test ---")

successful = 0
failed = 0

for case in converted_logs[:100]:
    try:
        features = extract_control_flow_features(case, dfg)

        assert all(
            isinstance(value, (int, float))
            for value in features.values()
        )

        successful += 1

    except Exception as exc:
        failed += 1
        print(f"FAILED case {case['case_id']}: {exc}")

print(f"Cases tested: {100}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")

if failed == 0:
    print("Control-flow feature extraction test PASSED.")
else:
    print("Control-flow feature extraction test FAILED.")

print("\n--- Full feature extraction test ---")

dfg = prune_dfg(build_dfg(converted_logs), min_freq=2)

X, y, subtype = extract_features(converted_logs, dfg)

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Subtypes shape: {subtype.shape}")

print("\n--- Feature columns ---")
for column in X.columns:
    print(f"{column}: {X[column].dtype}")

print("\n--- Missing values ---")
print(X.isna().sum())

print("\n--- Label distribution ---")
print(y.value_counts())

assert len(X) == len(converted_logs)
assert len(y) == len(converted_logs)
assert len(subtype) == len(converted_logs)
assert X.select_dtypes(exclude="number").empty
assert X.notna().all().all()

print("\nFull feature extraction test PASSED.")
