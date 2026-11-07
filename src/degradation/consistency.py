import copy
import random


UNKNOWN_RESOURCE = "unknown"


def inject_consistency_issues(
    logs: list[dict],
    level: float,
    variant: str = "resource_mismatch",
    seed: int = 42
) -> list[dict]:
    rng = random.Random(seed)
    corrupted = copy.deepcopy(logs)

    # Collect valid resource identifiers from the dataset.
    available_resources = {
        event.get("resource")
        for case in corrupted
        for event in case["events"]
        if event.get("resource") is not None
        and event.get("resource") != UNKNOWN_RESOURCE
    }

    n_corrupted = int(len(corrupted) * level)
    selected_indices = rng.sample(
        range(len(corrupted)),
        n_corrupted
    )

    for idx in selected_indices:
        case = corrupted[idx]

        modified = False

        if variant in {"resource_mismatch", "default"}:
            modified = _inject_resource_mismatch(
                case,
                available_resources,
                rng
            )

        elif variant == "mixed_resource_format":
            modified = _inject_mixed_resource_format(
                case,
                rng
            )

        else:
            raise ValueError(
                f"Unknown consistency variant: {variant}"
            )

        if modified:
            case["quality_issue"] = "consistency"
            case["quality_variant"] = variant

    return corrupted


def _inject_resource_mismatch(
    case: dict,
    available_resources: set,
    rng: random.Random
) -> bool:
    eligible_events = [
        event for event in case["events"]
        if event.get("resource") is not None
        and event.get("resource") != UNKNOWN_RESOURCE
    ]

    if not eligible_events:
        return False

    event = rng.choice(eligible_events)
    current_resource = event["resource"]

    alternatives = [
        resource for resource in available_resources
        if resource != current_resource
    ]

    if not alternatives:
        return False

    event["resource"] = rng.choice(alternatives)

    return True


def _inject_mixed_resource_format(
    case: dict,
    rng: random.Random
) -> bool:
    eligible_events = [
        event for event in case["events"]
        if isinstance(event.get("resource"), str)
        and event["resource"] != UNKNOWN_RESOURCE
    ]

    if not eligible_events:
        return False

    event = rng.choice(eligible_events)
    resource = event["resource"]

    if resource.islower():
        event["resource"] = resource.upper()
    elif resource.isupper():
        event["resource"] = resource.lower()
    else:
        event["resource"] = resource.swapcase()

    return True
