from src.process_mining.dfg import (
    wrong_order_ratio,
    unknown_edges_count,
    rare_edges_count,
    mean_edge_frequency,
    min_edge_frequency,
    dfg_path_log_likelihood,
)


EXPECTED_ACTIVITIES = {"A", "B", "C", "D", "E"}


def extract_control_flow_features(case: dict, dfg: dict) -> dict:
    activities = [e["activity"] for e in case["events"]]
    activity_set = set(activities)

    return {
        "has_loop": int(_has_loop(activities)),
        "has_skip": int(_has_skip(activity_set)),
        "missing_activities": len(EXPECTED_ACTIVITIES - activity_set),
        "extra_activities": len(activity_set - EXPECTED_ACTIVITIES),

        "unknown_edges_count": unknown_edges_count(case, dfg),
        "unknown_edges_ratio": wrong_order_ratio(case, dfg),
        "rare_edges_count": rare_edges_count(case, dfg),
        "mean_edge_frequency": mean_edge_frequency(case, dfg),
        "min_edge_frequency": min_edge_frequency(case, dfg),
        "dfg_path_log_likelihood": dfg_path_log_likelihood(case, dfg),
    }


def _has_loop(activities: list[str]) -> bool:
    return any(a == b for a, b in zip(activities, activities[1:]))


def _has_skip(activity_set: set[str]) -> bool:
    return "C" not in activity_set or "D" not in activity_set
