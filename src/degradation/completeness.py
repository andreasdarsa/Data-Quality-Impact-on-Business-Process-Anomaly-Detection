import copy
import random


def inject_missing_events(
    logs: list[dict],
    level: float,
    variant: str = "random",
    seed: int = 42,
    min_events: int = 2
) -> list[dict]:
    rng = random.Random(seed)
    corrupted = copy.deepcopy(logs)

    n_corrupted = int(len(corrupted) * level)
    selected_indices = rng.sample(range(len(corrupted)), n_corrupted)

    for idx in selected_indices:
        case = corrupted[idx]
        events = case["events"]

        if len(events) <= min_events:
            continue

        remove_idx = _select_event_to_remove(events, variant, rng)
        events.pop(remove_idx)

        _recompute_case_duration(case)

        case["quality_issue"] = "completeness"
        case["quality_variant"] = variant

    return corrupted


def _select_event_to_remove(
    events: list[dict],
    variant: str,
    rng: random.Random
) -> int:
    if variant == "targeted":
        target_activities = {"C", "D"}
        candidates = [
            i for i, event in enumerate(events)
            if event["activity"] in target_activities
        ]

        if candidates:
            return rng.choice(candidates)

    return rng.randrange(len(events))


def _recompute_case_duration(case: dict) -> None:
    case["total_duration"] = sum(event["duration"] for event in case["events"])
