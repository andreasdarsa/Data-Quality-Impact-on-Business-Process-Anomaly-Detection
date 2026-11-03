from datetime import datetime
from pathlib import Path
from typing import Any
from collections import defaultdict, deque

from src.data.loader import load_json


def load_binet_log(path: str | Path) -> dict:
    """Load a BINet event log and validate its basic structure."""
    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError("Expected a BINet event log to be a JSON object.")

    if not isinstance(data.get("cases"), list):
        raise ValueError("Invalid BINet event log: 'cases' must be a list.")

    return data


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO timestamp, returning None when unavailable."""
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _get_label(case: dict) -> tuple[bool, str]:
    """Convert a BINet label into an anomaly flag and subtype."""
    attributes = case.get("attributes", {})
    label = attributes.get("label", "normal")

    if label == "normal" or label is None:
        return False, "normal"

    if isinstance(label, dict):
        subtype = label.get("anomaly", "unknown")
        return True, str(subtype)

    # Some BINet exports represent dictionary labels as strings.
    if isinstance(label, str):
        if label.strip().lower() == "normal":
            return False, "normal"

        return True, _parse_string_label(label)

    return True, "unknown"


def _parse_string_label(label: str) -> str:
    """Extract the anomaly type from a stringified Python dictionary."""
    import ast

    try:
        parsed = ast.literal_eval(label)
    except (ValueError, SyntaxError):
        return "unknown"

    if isinstance(parsed, dict):
        return str(parsed.get("anomaly", "unknown"))

    return "unknown"


def _extract_events(case: dict) -> list[dict]:
    """
    Convert BINet lifecycle events into activity-level events.

    Only COMPLETE transitions are retained.
    START transitions are used to infer activity durations.
    START/COMPLETE pairs are matched using FIFO per activity.
    """
    raw_events = case.get("events", [])

    if not isinstance(raw_events, list):
        raise ValueError(
            f"Case {case.get('id', '<unknown>')} has invalid events."
        )

    parsed_events = []

    for index, raw_event in enumerate(raw_events):
        attributes = raw_event.get("attributes", {})

        activity = attributes.get("concept:name")
        if activity is None:
            activity = raw_event.get("name", "unknown")

        transition = attributes.get("lifecycle:transition")

        timestamp = (
            raw_event.get("timestamp")
            or attributes.get("time:timestamp")
        )

        timestamp_dt = _parse_timestamp(timestamp)

        parsed_events.append({
            "original_index": index,
            "activity": str(activity),
            "transition": transition,
            "timestamp_dt": timestamp_dt,
            "resource": attributes.get("org:resource", "unknown"),
        })

    # Process lifecycle events chronologically.
    parsed_events.sort(
        key=lambda event: (
            event["timestamp_dt"] is None,
            event["timestamp_dt"] or datetime.max,
            event["original_index"],
        )
    )

    pending_starts = defaultdict(deque)
    completed_events = []

    for event in parsed_events:
        activity = event["activity"]
        transition = event["transition"]
        timestamp_dt = event["timestamp_dt"]

        # Store activity start timestamps.
        if transition == "START":
            if timestamp_dt is not None:
                pending_starts[activity].append(timestamp_dt)

            continue

        # Retain only completed activities.
        if transition != "COMPLETE":
            continue

        start_dt = None
        duration = None

        if timestamp_dt is not None and pending_starts[activity]:
            candidate_start = pending_starts[activity].popleft()

            elapsed_seconds = (
                timestamp_dt - candidate_start
            ).total_seconds()

            if elapsed_seconds >= 0:
                start_dt = candidate_start
                duration = elapsed_seconds

        completed_events.append({
            "_completion_dt": timestamp_dt,
            "_original_index": event["original_index"],
            "activity": activity,
            "timestamp_start": (
                start_dt.isoformat() if start_dt else None
            ),
            "timestamp_end": (
                timestamp_dt.isoformat() if timestamp_dt else None
            ),
            "duration": duration,
            "resource": event["resource"],
        })

    # Order activities by their completion timestamps.
    completed_events.sort(
        key=lambda event: (
            event["_completion_dt"] is None,
            event["_completion_dt"] or datetime.max,
            event["_original_index"],
        )
    )

    # Remove internal sorting fields.
    return [
        {
            key: value
            for key, value in event.items()
            if not key.startswith("_")
        }
        for event in completed_events
    ]


def convert_binet_case(case: dict) -> dict:
    """Convert one BINet case to the project's canonical case schema."""
    attributes = case.get("attributes", {})

    if not isinstance(attributes, dict):
        attributes = {}

    events = _extract_events(case)

    # Calculate case duration using activity completion timestamps.
    timestamps = [
        _parse_timestamp(event["timestamp_end"])
        for event in events
    ]

    valid_timestamps = [
        timestamp for timestamp in timestamps
        if timestamp is not None
    ]

    if len(valid_timestamps) >= 2:
        total_duration = (
            max(valid_timestamps) - min(valid_timestamps)
        ).total_seconds()
    else:
        total_duration = 0.0

    is_anomaly, subtype = _get_label(case)

    amount_value = attributes.get("AMOUNT_REQ", 0)

    try:
        amount = (
            float(amount_value)
            if amount_value is not None
            else 0.0
        )
    except (TypeError, ValueError):
        amount = 0.0

    return {
        "case_id": str(
            case.get("id", attributes.get("concept:name", "unknown"))
        ),
        "events": events,
        "amount": amount,
        "priority": 0,
        "total_duration": total_duration,
        "is_anomaly": is_anomaly,
        "subtype": subtype,
    }


def convert_binet_log(data: dict) -> list[dict]:
    """Convert all cases in a BINet log to the project's canonical schema."""
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("Invalid BINet log: expected a dictionary with a 'cases' list.")

    converted = []

    for case in data["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Invalid BINet log: every case must be a dictionary.")

        converted_case = convert_binet_case(case)

        if not converted_case["events"]:
            raise ValueError(
                f"Case {converted_case['case_id']} has no events."
            )

        converted.append(converted_case)

    return converted


def load_and_convert_binet_log(path: str | Path) -> list[dict]:
    """Load a BINet JSON file and convert it to the project's log schema."""
    return convert_binet_log(load_binet_log(path))