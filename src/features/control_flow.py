from src.process_mining.dfg import wrong_order_ratio


EXPECTED_ACTIVITIES = {"A", "B", "C", "D", "E"}


def extract_control_flow_features(case: dict, dfg: dict) -> dict:
    activities = [e["activity"] for e in case["events"]]
    activity_set = set(activities)

    return {
        "has_loop": int(_has_loop(activities)),
        "has_skip": int(_has_skip(activity_set)),
        "missing_activities": len(EXPECTED_ACTIVITIES - activity_set),
        "extra_activities": len(activity_set - EXPECTED_ACTIVITIES),
        "wrong_order_ratio": wrong_order_ratio(case, dfg),
    }


def _has_loop(activities: list[str]) -> bool:
    return any(a == b for a, b in zip(activities, activities[1:]))


def _has_skip(activity_set: set[str]) -> bool:
    return "C" not in activity_set or "D" not in activity_set
