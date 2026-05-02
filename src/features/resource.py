def extract_resource_features(case: dict) -> dict:
    events = case["events"]

    resources = [e.get("resource", "unknown") for e in events]

    senior_count = sum(1 for r in resources if r == "senior")
    junior_count = sum(1 for r in resources if r == "junior")

    total = len(resources)

    return {
        "unique_resources": len(set(resources)),
        "senior_ratio": senior_count / total if total else 0,
        "junior_ratio": junior_count / total if total else 0,
    }
