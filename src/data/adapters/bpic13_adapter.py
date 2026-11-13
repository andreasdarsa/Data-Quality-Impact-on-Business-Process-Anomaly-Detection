
"""Adapter for the BPIC 2013 (closed problems) event log."""

from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.loader import load_json


def load_bpic13_log(path: str | Path) -> dict:
    """Load a BPIC13 JSON log and validate its top-level structure."""
    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError("Expected a BPIC13 event log to be a JSON object.")

    if not isinstance(data.get("cases"), list):
        raise ValueError("Invalid BPIC13 event log: 'cases' must be a list.")

    return data


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO timestamp, returning None when unavailable."""
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _normalise_label(label: Any) -> tuple[bool, str]:
    """Return the binary anomaly flag and the anomaly subtype."""
    if label is None:
        return False, "normal"

    if isinstance(label, str):
        if label.strip().lower() == "normal":
            return False, "normal"
        return True, label.strip() or "unknown"

    if isinstance(label, dict):
        subtype = label.get("anomaly")
        return True, str(subtype) if subtype else "unknown"

    return True, "unknown"


def _extract_events(case: dict) -> list[dict]:
    """Convert BPIC13 events without sorting or dropping events."""
    raw_events = case.get("events", [])

    if not isinstance(raw_events, list):
        raise ValueError(
            f"Case {case.get('id', '<unknown>')} has invalid events."
        )

    converted = []

    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise ValueError(
                f"Case {case.get('id', '<unknown>')} contains "
                f"a non-object event at index {index}."
            )

        attributes = raw_event.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}

        activity = attributes.get("concept:name")

        if activity is None or not str(activity).strip():
            activity = raw_event.get("name")

        if activity is None or not str(activity).strip():
            activity = "unknown"

        timestamp_value = (
            raw_event.get("timestamp")
            or attributes.get("time:timestamp")
        )

        timestamp_dt = _parse_timestamp(timestamp_value)
        timestamp = (
            timestamp_dt.isoformat() if timestamp_dt else None
        )

        resource = attributes.get("org:resource")

        if resource is None or not str(resource).strip():
            resource = "unknown"
        else:
            resource = str(resource)

        converted.append({
            "activity": str(activity),
            "timestamp_start": None,
            "timestamp_end": timestamp,
            "duration": None,
            "resource": resource,
            "lifecycle_transition": attributes.get(
                "lifecycle:transition"
            ),
            "event_name": raw_event.get("name"),
            "attributes": attributes.copy(),
            "original_index": index,
        })

    return converted


def convert_bpic13_case(case: dict) -> dict:
    """Convert one BPIC13 case to the canonical project schema."""
    if not isinstance(case, dict):
        raise ValueError("Every BPIC13 case must be a dictionary.")

    attributes = case.get("attributes", {})

    if not isinstance(attributes, dict):
        attributes = {}

    events = _extract_events(case)

    if not events:
        raise ValueError(
            f"Case {case.get('id', '<unknown>')} has no events."
        )

    is_anomaly, subtype = _normalise_label(
        attributes.get("label", "normal")
    )

    timestamps = [
        timestamp
        for timestamp in (
            _parse_timestamp(event["timestamp_end"])
            for event in events
        )
        if timestamp is not None
    ]

    total_duration = (
        (max(timestamps) - min(timestamps)).total_seconds()
        if len(timestamps) >= 2
        else 0.0
    )

    case_id = case.get("id", attributes.get("concept:name"))

    if case_id is None:
        raise ValueError(
            "BPIC13 case is missing both 'id' and case concept:name."
        )

    return {
        "case_id": str(case_id),
        "events": events,
        "amount": 0.0,
        "priority": 0,
        "total_duration": total_duration,
        "is_anomaly": is_anomaly,
        "subtype": subtype,
    }


def convert_bpic13_log(data: dict) -> list[dict]:
    """Convert all cases in a BPIC13 log."""
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(
            "Invalid BPIC13 log: expected a dictionary with a 'cases' list."
        )

    converted = [
        convert_bpic13_case(case)
        for case in data["cases"]
    ]

    case_ids = [case["case_id"] for case in converted]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("BPIC13 log contains duplicate case IDs.")

    return converted


def load_and_convert_bpic13_log(path: str | Path) -> list[dict]:
    """Load and convert a BPIC13 JSON event log."""
    return convert_bpic13_log(load_bpic13_log(path))
