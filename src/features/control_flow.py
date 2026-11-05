from src.process_mining.dfg import (
    wrong_order_ratio,
    unknown_edges_count,
    rare_edges_count,
    mean_edge_frequency,
    min_edge_frequency,
    dfg_path_log_likelihood,
)


def extract_control_flow_features(case: dict, dfg: dict) -> dict:
    activities = [event["activity"] for event in case["events"]]

    return {
        "has_loop": int(_has_loop(activities)),
        "unique_activities": len(set(activities)),
        "trace_length": len(activities),

        "unknown_edges_count": unknown_edges_count(case, dfg),
        "unknown_edges_ratio": wrong_order_ratio(case, dfg),
        "rare_edges_count": rare_edges_count(case, dfg),
        "mean_edge_frequency": mean_edge_frequency(case, dfg),
        "min_edge_frequency": min_edge_frequency(case, dfg),
        "dfg_path_log_likelihood": dfg_path_log_likelihood(case, dfg),
    }


def _has_loop(activities: list[str]) -> bool:
    return any(
        current == previous
        for previous, current in zip(activities, activities[1:])
    )