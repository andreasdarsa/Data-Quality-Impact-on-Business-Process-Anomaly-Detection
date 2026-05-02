import copy
import random


VALID_RESOURCES = ["junior", "senior"]


def inject_consistency_issues(
    logs: list[dict],
    level: float,
    variant: str = "resource_mismatch",
    seed: int = 42
) -> list[dict]:
    rng = random.Random(seed)
    corrupted = copy.deepcopy(logs)

    n_corrupted = int(len(corrupted) * level)
    selected_indices = rng.sample(range(len(corrupted)), n_corrupted)

    for idx in selected_indices:
        case = corrupted[idx]

        if variant in {"resource_mismatch", "default"}:
            _inject_resource_mismatch(case, rng)

        elif variant == "priority_duration_conflict":
            _inject_priority_duration_conflict(case)

        elif variant == "mixed_resource_format":
            _inject_mixed_resource_format(case)

        else:
            raise ValueError(f"Unknown consistency variant: {variant}")

        case["quality_issue"] = "consistency"
        case["quality_variant"] = variant

    return corrupted


def _inject_resource_mismatch(case: dict, rng: random.Random) -> None:
    if not case["events"]:
        return

    event = rng.choice(case["events"])

    if event["resource"] == "senior":
        event["resource"] = "junior"
    else:
        event["resource"] = "senior"


def _inject_priority_duration_conflict(case: dict) -> None:
    case["priority"] = "high"
    case["total_duration"] = case["total_duration"] * 2


def _inject_mixed_resource_format(case: dict) -> None:
    for event in case["events"]:
        if event["resource"] == "senior":
            event["resource"] = "Senior"
        elif event["resource"] == "junior":
            event["resource"] = "JUNIOR"
