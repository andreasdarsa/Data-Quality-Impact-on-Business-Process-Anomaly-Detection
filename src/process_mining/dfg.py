from collections import Counter
from typing import Dict, List, Tuple
import math


Edge = Tuple[str, str]
DFG = Dict[Edge, int]


def extract_trace(case: dict) -> List[str]:
    """
    Returns the activity sequence of a case.
    Assumes events are already in the order stored in the log.
    """
    return [event["activity"] for event in case["events"]]


def build_dfg(logs: List[dict]) -> DFG:
    """
    Builds a Directly-Follows Graph from a list of cases.

    Example:
        A -> B -> C creates edges:
        (A, B), (B, C)
    """
    edge_counts = Counter()

    for case in logs:
        trace = extract_trace(case)

        for a, b in zip(trace, trace[1:]):
            edge_counts[(a, b)] += 1

    return dict(edge_counts)


def prune_dfg(dfg: DFG, min_freq: int = 2) -> DFG:
    """
    Removes low-frequency edges from the DFG.
    Useful because rare edges may come from anomalies or corrupted logs.
    """
    return {
        edge: count
        for edge, count in dfg.items()
        if count >= min_freq
    }


def normalize_dfg(dfg: DFG) -> Dict[Edge, float]:
    """
    Converts edge counts into relative frequencies.
    Useful for analysis/debugging.
    """
    total = sum(dfg.values())

    if total == 0:
        return {}

    return {
        edge: count / total
        for edge, count in dfg.items()
    }


def get_allowed_edges(dfg: DFG) -> set[Edge]:
    """
    Returns the set of allowed directly-follows relations.
    Used later by control-flow feature extraction.
    """
    return set(dfg.keys())


def edge_exists(dfg: DFG, source: str, target: str) -> bool:
    """
    Checks whether a directly-follows edge exists in the DFG.
    """
    return (source, target) in dfg


def get_trace_edges(case: dict) -> List[Edge]:
    """
    Returns all directly-follows edges of a case.
    """
    trace = extract_trace(case)
    return list(zip(trace, trace[1:]))


def count_unknown_edges(case: dict, dfg: DFG) -> int:
    """
    Counts how many directly-follows edges in a case
    do not exist in the reference DFG.
    """
    allowed_edges = get_allowed_edges(dfg)
    trace_edges = get_trace_edges(case)

    return sum(
        1 for edge in trace_edges
        if edge not in allowed_edges
    )


def wrong_order_ratio(case: dict, dfg: DFG) -> float:
    """
    Ratio of unknown directly-follows edges in a case.

    Example:
        trace has 4 edges, 1 unknown edge
        wrong_order_ratio = 0.25
    """
    trace_edges = get_trace_edges(case)

    if not trace_edges:
        return 0.0

    unknown = count_unknown_edges(case, dfg)
    return unknown / len(trace_edges)


def get_edge_frequency(dfg: dict, edge: tuple[str, str]) -> int:
    return dfg.get(edge, 0)


def get_trace_edge_frequencies(case: dict, dfg: dict) -> list[int]:
    trace_edges = get_trace_edges(case)
    return [get_edge_frequency(dfg, edge) for edge in trace_edges]


def unknown_edges_count(case: dict, dfg: dict) -> int:
    return count_unknown_edges(case, dfg)


def mean_edge_frequency(case: dict, dfg: dict) -> float:
    freqs = get_trace_edge_frequencies(case, dfg)
    return sum(freqs) / len(freqs) if freqs else 0.0


def min_edge_frequency(case: dict, dfg: dict) -> float:
    freqs = get_trace_edge_frequencies(case, dfg)
    return min(freqs) if freqs else 0.0


def rare_edges_count(case: dict, dfg: dict, rare_threshold: int = 5) -> int:
    freqs = get_trace_edge_frequencies(case, dfg)

    return sum(
        1 for f in freqs
        if 0 < f <= rare_threshold
    )


def dfg_path_log_likelihood(case: dict, dfg: dict, smoothing: float = 1e-6) -> float:
    trace_edges = get_trace_edges(case)

    if not trace_edges:
        return 0.0

    total = sum(dfg.values())

    if total == 0:
        return 0.0

    score = 0.0

    for edge in trace_edges:
        probability = dfg.get(edge, 0) / total
        probability = max(probability, smoothing)
        score += math.log(probability)

    return score
