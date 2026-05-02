def extract_basic_features(case: dict) -> dict:
    events = case["events"]

    return {
        "num_events": len(events),
        "unique_activities": len(set(e["activity"] for e in events)),
        "total_duration": case.get("total_duration", 0),
        "amount": case.get("amount", 0),
    }
