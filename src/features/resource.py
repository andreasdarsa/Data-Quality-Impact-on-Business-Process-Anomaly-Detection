from collections import Counter


def extract_resource_features(case: dict) -> dict:
    events = case["events"]

    resources = [
        event.get("resource", "unknown")
        for event in events
    ]

    total = len(resources)

    if total == 0:
        return {
            "unique_resources": 0,
            "unknown_resource_ratio": 0.0,
            "dominant_resource_ratio": 0.0,
            "resource_diversity": 0.0,
        }

    resource_counts = Counter(
        resource
        for resource in resources
        if resource not in (None, "", "unknown")
    )

    known_resources = sum(resource_counts.values())

    unique_resources = len(resource_counts)

    unknown_count = total - known_resources

    dominant_count = (
        max(resource_counts.values())
        if resource_counts
        else 0
    )

    return {
        "unique_resources": unique_resources,
        "unknown_resource_ratio": unknown_count / total,
        "dominant_resource_ratio": dominant_count / total,
        "resource_diversity": unique_resources / known_resources
        if known_resources else 0.0,
    }