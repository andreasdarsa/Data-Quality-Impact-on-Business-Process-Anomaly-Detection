REQUIRED_CASE_FIELDS = {
    "case_id",
    "events",
    "amount",
    "priority",
    "total_duration",
    "is_anomaly",
    "subtype",
}

REQUIRED_EVENT_FIELDS = {
    "activity",
    "timestamp_start",
    "timestamp_end",
    "duration",
    "resource",
}


def validate_logs(logs: list[dict]) -> None:
    if not isinstance(logs, list):
        raise ValueError("Logs must be a list of cases.")

    for case in logs:
        validate_case(case)


def validate_case(case: dict) -> None:
    missing = REQUIRED_CASE_FIELDS - set(case.keys())

    if missing:
        raise ValueError(
            f"Case {case.get('case_id', '<unknown>')} is missing fields: {missing}"
        )

    if not isinstance(case["events"], list):
        raise ValueError(f"Case {case['case_id']} has invalid events field.")

    if len(case["events"]) == 0:
        raise ValueError(f"Case {case['case_id']} has no events.")

    for event in case["events"]:
        validate_event(event, case_id=case["case_id"])


def validate_event(event: dict, case_id: str | None = None) -> None:
    missing = REQUIRED_EVENT_FIELDS - set(event.keys())

    if missing:
        raise ValueError(
            f"Event in case {case_id or '<unknown>'} is missing fields: {missing}"
        )
