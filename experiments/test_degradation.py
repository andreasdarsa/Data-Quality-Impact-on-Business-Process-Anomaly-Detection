from copy import deepcopy

from src.data.loader import load_logs
from src.data.schema import validate_logs
from src.degradation import apply_degradation
from src.process_mining import build_dfg, prune_dfg
from src.features import extract_features
from src.models import fit_all_detectors, evaluate_all_detectors
from src.evaluation import evaluate_detector_results

from src.utils.paths import PROCESSED_DATA_DIR

from experiments.configs import (
    THRESHOLD_PERCENTILE,
    get_binet_train_filename,
    get_binet_test_filename,
    get_degradation_configs,
)

from datetime import datetime


def parse_timestamp(value):
    """Convert an ISO timestamp into a datetime object."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    return None


def timestamp_difference_minutes(old_value, new_value):
    """Return the absolute timestamp difference in minutes."""
    old_dt = parse_timestamp(old_value)
    new_dt = parse_timestamp(new_value)

    if old_dt is None or new_dt is None:
        return None

    # Handle timestamps with different timezone awareness.
    if (old_dt.tzinfo is None) != (new_dt.tzinfo is None):
        return None

    return abs((new_dt - old_dt).total_seconds()) / 60


def audit_degradation(
    original_logs: list[dict],
    degraded_logs: list[dict],
    dimension: str,
) -> dict:

    changed_cases = 0
    changed_values = 0

    durations_modified = 0
    timestamps_modified = 0
    resources_modified = 0

    timestamp_differences = []

    original_events = sum(
        len(case["events"]) for case in original_logs
    )

    degraded_events = sum(
        len(case["events"]) for case in degraded_logs
    )

    for original_case, degraded_case in zip(
        original_logs,
        degraded_logs
    ):

        case_changed = False

        original_case_events = original_case["events"]
        degraded_case_events = degraded_case["events"]

        if dimension == "completeness":

            removed = (
                len(original_case_events)
                - len(degraded_case_events)
            )

            if removed > 0:
                case_changed = True
                changed_values += removed

        else:

            for old_event, new_event in zip(
                original_case_events,
                degraded_case_events
            ):

                if dimension == "accuracy":

                    if old_event.get("duration") != new_event.get("duration"):
                        case_changed = True
                        durations_modified += 1

                    timestamp_fields = [
                        "timestamp_start",
                        "timestamp_end",
                    ]

                elif dimension == "consistency":

                    if old_event.get("resource") != new_event.get("resource"):
                        case_changed = True
                        resources_modified += 1

                    timestamp_fields = []

                elif dimension == "timeliness":

                    timestamp_fields = [
                        "timestamp_start",
                        "timestamp_end",
                    ]

                else:
                    raise ValueError(
                        f"Unknown degradation dimension: {dimension}"
                    )

                for field in timestamp_fields:

                    old_value = old_event.get(field)
                    new_value = new_event.get(field)

                    if old_value != new_value:

                        case_changed = True
                        timestamps_modified += 1

                        difference = timestamp_difference_minutes(
                            old_value,
                            new_value
                        )

                        if difference is not None:
                            timestamp_differences.append(difference)

        if case_changed:
            changed_cases += 1

    if dimension == "completeness":
        changed_values = original_events - degraded_events

    elif dimension == "accuracy":
        changed_values = (
            durations_modified + timestamps_modified
        )

    elif dimension == "consistency":
        changed_values = resources_modified

    elif dimension == "timeliness":
        changed_values = timestamps_modified

    print("\n--- Degradation Audit ---")
    print(f"Dimension: {dimension}")
    print(
        f"Cases modified: "
        f"{changed_cases}/{len(original_logs)}"
    )
    print(f"Values modified: {changed_values}")

    if dimension == "completeness":
        print(f"Original events: {original_events}")
        print(f"Remaining events: {degraded_events}")
        print(f"Events removed: {original_events - degraded_events}")

    if dimension == "accuracy":
        print(f"Durations modified: {durations_modified}")
        print(f"Timestamps modified: {timestamps_modified}")

    if dimension == "consistency":
        print(f"Resources modified: {resources_modified}")

    if dimension in ["accuracy", "timeliness"]:

        if timestamp_differences:
            print(
                "Average absolute timestamp shift: "
                f"{sum(timestamp_differences) / len(timestamp_differences):.2f} minutes"
            )
            print(
                "Maximum absolute timestamp shift: "
                f"{max(timestamp_differences):.2f} minutes"
            )
        else:
            print("No measurable timestamp changes.")

    print("--------------------------\n")

    return {
        "dimension": dimension,
        "cases_modified": changed_cases,
        "values_modified": changed_values,
        "durations_modified": durations_modified,
        "timestamps_modified": timestamps_modified,
        "resources_modified": resources_modified,
        "events_removed": original_events - degraded_events,
        "average_timestamp_shift_minutes": (
            sum(timestamp_differences) / len(timestamp_differences)
            if timestamp_differences else 0
        ),
        "max_timestamp_shift_minutes": (
            max(timestamp_differences)
            if timestamp_differences else 0
        ),
    }

def audit_data_integrity(logs: list[dict]) -> dict:

    empty_cases = 0
    invalid_intervals = 0
    chronological_inversions = 0
    checked_intervals = 0
    checked_event_pairs = 0

    for case in logs:

        events = case["events"]

        # 1. Check for empty cases.
        if len(events) == 0:
            empty_cases += 1
            continue

        # 2. Check event-level timestamp validity.
        for event in events:

            start = parse_timestamp(
                event.get("timestamp_start")
            )

            end = parse_timestamp(
                event.get("timestamp_end")
            )

            if start is not None and end is not None:

                if (
                    (start.tzinfo is None) == (end.tzinfo is None)
                ):
                    checked_intervals += 1

                    if start > end:
                        invalid_intervals += 1

        # 3. Check chronological order of completion timestamps.
        valid_completion_times = []

        for event in events:

            completion = parse_timestamp(
                event.get("timestamp_end")
            )

            if completion is not None:
                valid_completion_times.append(completion)

        for previous, current in zip(
            valid_completion_times,
            valid_completion_times[1:]
        ):

            if (
                (previous.tzinfo is None)
                != (current.tzinfo is None)
            ):
                continue

            checked_event_pairs += 1

            if current < previous:
                chronological_inversions += 1

    result = {
        "empty_cases": empty_cases,
        "invalid_intervals": invalid_intervals,
        "chronological_inversions": chronological_inversions,
        "checked_intervals": checked_intervals,
        "checked_event_pairs": checked_event_pairs,
    }

    print("\n--- Data Integrity Audit ---")
    print(f"Empty cases: {empty_cases}")
    print(
        f"Invalid event intervals: "
        f"{invalid_intervals}/{checked_intervals}"
    )
    print(
        f"Chronological inversions: "
        f"{chronological_inversions}/{checked_event_pairs}"
    )
    print("----------------------------\n")

    return result

def main():

    split_seed = 42
    degradation_seed = 101

    print("Loading BINet datasets...")

    train_logs = load_logs(
        PROCESSED_DATA_DIR / get_binet_train_filename(split_seed)
    )

    test_logs = load_logs(
        PROCESSED_DATA_DIR / get_binet_test_filename(split_seed)
    )

    validate_logs(train_logs)
    validate_logs(test_logs)

    print(
        f"Training cases: {len(train_logs)} | "
        f"Test cases: {len(test_logs)}"
    )

    # Build DFG using normal training cases only.
    normal_train_logs = [
        case for case in train_logs
        if not case["is_anomaly"]
    ]

    dfg = build_dfg(normal_train_logs)
    dfg = prune_dfg(dfg, min_freq=2)

    # Extract features from all training cases.
    X_train, _, _ = extract_features(train_logs, dfg)

    print("Fitting detectors...")

    fitted_models = fit_all_detectors(
        X_train=X_train,
        threshold_percentile=THRESHOLD_PERCENTILE,
        seed=split_seed,
    )

    # Get all degradation configurations.
    configs = get_degradation_configs(degradation_seed)

    expected_configs = 12

    if len(configs) != expected_configs:
        raise ValueError(
            f"Expected {expected_configs} configurations, "
            f"got {len(configs)}"
        )

    print(f"\nTotal degradation tests: {len(configs)}")

    # Keep original test data unchanged.
    # Keep original test data unchanged.
    original_test_logs = deepcopy(test_logs)

    audit_results = []

    for config in configs:

        print(
            f"\n{'=' * 60}\n"
            f"Testing {config.dimension} "
            f"(level={config.level}, seed={config.seed})"
            f"\n{'=' * 60}"
        )

        # Apply degradation independently to the original test set.
        degraded_logs = apply_degradation(test_logs, config)

        # 1. Determinism test.
        second_run = apply_degradation(test_logs, config)

        assert degraded_logs == second_run, (
            f"Non-deterministic output detected: "
            f"{config.dimension}, level={config.level}, "
            f"seed={config.seed}"
        )

        print("Determinism: PASSED")

        # 2. Data integrity test.
        integrity = audit_data_integrity(degraded_logs)

        assert integrity["empty_cases"] == 0, (
            f"Empty cases detected in {config.dimension}!"
        )

        assert integrity["invalid_intervals"] == 0, (
            f"Invalid event intervals detected in {config.dimension}!"
        )

        print("Data integrity: PASSED")

        # 1. Audit actual data modifications.
        audit_result = audit_degradation(
            original_logs=original_test_logs,
            degraded_logs=degraded_logs,
            dimension=config.dimension,
        )

        audit_result["level"] = config.level
        audit_result["seed"] = config.seed

        audit_results.append(audit_result)

        # 2. Schema validation.
        validate_logs(degraded_logs)

        # 3. Verify labels remain unchanged.
        original_labels = [
            (case["is_anomaly"], case["subtype"])
            for case in original_test_logs
        ]

        degraded_labels = [
            (case["is_anomaly"], case["subtype"])
            for case in degraded_logs
        ]

        assert original_labels == degraded_labels, (
            f"Labels changed during {config.dimension} "
            f"degradation at level {config.level}!"
        )

        # 4. Verify original test logs remain unchanged.
        assert test_logs == original_test_logs, (
            f"Original test data was modified by "
            f"{config.dimension}!"
        )

        # 5. Feature extraction.
        X_test, y_test, test_subtype = extract_features(
            degraded_logs,
            dfg
        )

        assert X_test.shape[0] == len(degraded_logs)
        assert len(y_test) == len(degraded_logs)
        assert len(test_subtype) == len(degraded_logs)

        # 6. Evaluate using the same fitted detectors.
        detector_outputs = evaluate_all_detectors(
            fitted_models=fitted_models,
            X_test=X_test,
        )

        metrics = evaluate_detector_results(
            y_true=y_test,
            detector_results=detector_outputs,
        )

        # 7. Print metrics.
        print(f"Cases: {len(degraded_logs)}")
        print(f"Features: {X_test.shape[1]}")

        for detector_name, detector_metrics in metrics.items():
            print(
                f"{detector_name}: "
                f"Precision={detector_metrics['precision']:.4f}, "
                f"Recall={detector_metrics['recall']:.4f}, "
                f"F1={detector_metrics['f1']:.4f}"
            )

        print(
            f"{config.dimension} "
            f"(level={config.level}): PASSED"
        )


    # 8. Print degradation audit summary.
    print("\n\n========== DEGRADATION AUDIT SUMMARY ==========")

    for dimension in [
        "completeness",
        "accuracy",
        "consistency",
        "timeliness",
    ]:

        print(f"\n{dimension.upper()}")

        for result in audit_results:

            if result["dimension"] != dimension:
                continue

            print(
                f"Level {result['level']:.0%}: "
                f"{result['cases_modified']} cases modified | "
                f"{result['values_modified']} values modified"
            )

    print("\nAll 12 degradation smoke tests passed!")


if __name__ == "__main__":
    main()
