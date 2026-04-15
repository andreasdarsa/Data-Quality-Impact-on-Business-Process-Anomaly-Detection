import json
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def log_json_to_df(data: list) -> pd.DataFrame:
    rows = []
    for case in data:
        case_id = case["case_id"]
        is_anomaly = case.get("is_anomaly", 0)
        for event in case["events"]:
            rows.append({
                "case_id": case_id,
                "activity": event["activity"],
                "start_timestamp": pd.to_datetime(event["timestamp_start"]),
                "end_timestamp": pd.to_datetime(event["timestamp_end"]),
                "resource": event["resource"],
                "is_anomaly": is_anomaly,
                "subtype": case.get("subtype", "normal"),
                "anomaly_type": case.get("anomaly_type", "normal")
            })

    df = pd.DataFrame(rows)

    df["activity"] = df["activity"].astype(str)
    df["resource"] = df["resource"].astype(str)
    df["case_id"] = df["case_id"].astype(str)
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])
    df["end_timestamp"] = pd.to_datetime(df["end_timestamp"])
    df["is_anomaly"] = df["is_anomaly"].astype(int)

    df = df.rename(columns={
        'case_id': 'case:concept:name',
        'start_timestamp': 'time:timestamp',
        'activity': 'concept:name'
    })

    return df


def filter_normal_cases(df: pd.DataFrame) -> pd.DataFrame:
    # κρατάμε μόνο cases με is_anomaly = 0
    normal_cases = df[df["is_anomaly"] == 0]["case:concept:name"].unique()
    df_normal = df[df["case:concept:name"].isin(normal_cases)]
    return df_normal


def draw_dfg(dfg: dict) -> None:
    G = nx.DiGraph()
    for (src, tgt), cnt in dfg.items():
        G.add_edge(src, tgt, weight=cnt)

    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(G, pos, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)

    for (u, v) in G.edges():
        if (v, u) in G.edges() and u != v:
            # υπάρχει και αντίθετη ακμή → κάνε καμπύλη
            rad = 0.2 if u < v else -0.2
        else:
            rad = 0.0

        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            connectionstyle=f'arc3,rad={rad}',
            arrows=True
        )

    edge_labels = nx.get_edge_attributes(G, 'weight')

    for (u, v), label in edge_labels.items():
        if (v, u) in G.edges() and u != v:
            rad = 0.2 if u < v else -0.2
        else:
            rad = 0.0

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={(u, v): label},
            label_pos=0.5,
            rotate=False
        )

    plt.savefig("bpm/img/dfg.png", dpi=300)
    plt.title("Directly-Follows Graph (DFG)")
    plt.show()


def extract_case_features(df, dfg):
    cases = []

    for case_id, group in df.groupby("case:concept:name"):

        group = group.sort_values("time:timestamp")

        durations = (group["end_timestamp"] - group["time:timestamp"]).dt.total_seconds()
        durations = durations.fillna(0).values
        activities = group["concept:name"].values
        resources = group["resource"].values

        # --- Basic ---
        total_duration = durations.sum()
        num_events = len(group)
        unique_activities = len(set(activities))

        # --- Duration stats ---
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        min_duration = np.min(durations)
        max_duration = np.max(durations)

        # --- Gaps ---
        gaps = []
        timestamps_start = pd.to_datetime(group["time:timestamp"])
        timestamps_end = pd.to_datetime(group["end_timestamp"])

        for i in range(1, len(group)):
            gap = (timestamps_start.iloc[i] - timestamps_end.iloc[i - 1]).total_seconds() / 60
            gaps.append(gap)

        avg_gap = np.mean(gaps) if gaps else 0
        std_gap = np.std(gaps) if gaps else 0

        # --- Resource features ---
        num_unique_resources = len(set(resources))
        senior_ratio = sum(1 for r in resources if r == "senior") / num_events
        junior_ratio = sum(1 for r in resources if r == "junior") / num_events

        # --- Control flow ---
        has_loop = int(len(activities) != len(set(activities)))
        has_skip = int("C" not in activities)

        # --- Simple deviation features ---
        expected_flow = ["A", "B", "C", "D", "E"]

        missing = len([a for a in expected_flow if a not in activities])
        extra = max(0, len(activities) - len(expected_flow))

        # Wrong order (very simple metric)
        wrong_order = 0
        for i in range(len(activities) - 1):
            if (activities[i], activities[i + 1]) not in dfg:
                wrong_order += 1

        wrong_order_ratio = wrong_order / (len(activities) - 1) if len(activities) > 1 else 0

        # --- Noise injection (break duplicates) ---
        noise = np.random.normal(0, 0.01)

        cases.append({
            "case_id": case_id,

            # basic
            "total_duration": total_duration + noise,
            "num_events": num_events,
            "unique_activities": unique_activities,

            # duration stats
            "mean_duration": mean_duration + noise,
            "std_duration": std_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,

            # temporal
            "avg_gap": avg_gap + noise,
            "std_gap": std_gap,

            # resources
            "num_unique_resources": num_unique_resources,
            "senior_ratio": senior_ratio,
            "junior_ratio": junior_ratio,

            # control flow
            "has_loop": has_loop,
            "has_skip": has_skip,

            # deviation
            "missing_activities": missing,
            "extra_activities": extra,
            "wrong_order_ratio": wrong_order_ratio,

            # label
            "is_anomaly": group["is_anomaly"].iloc[0],

            # anomaly subtype
            "subtype": group["subtype"].iloc[0] if "subtype" in group.columns else "normal"
        })

    return pd.DataFrame(cases)


def prune_dfg(dfg: dict, min_ratio: float = 0.01) -> dict:
    """
    Κρατάει μόνο edges που εμφανίζονται τουλάχιστον min_ratio του max frequency
    """
    if not dfg:
        return dfg

    max_freq = max(dfg.values())
    threshold = min_ratio * max_freq

    pruned_dfg = {
        (a, b): cnt
        for (a, b), cnt in dfg.items()
        if cnt >= threshold
    }

    return pruned_dfg


def encode_case_features(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = features_df.drop(columns=["case_id", "is_anomaly", "subtype"])
    y = features_df["is_anomaly"]
    subtypes = features_df["subtype"]

    return X, y, subtypes


def mine_process_model() -> tuple[pd.DataFrame, dict]:
    with open(f"bpm/data/json/baseline_event_log.json", "r") as f:
        data = json.load(f)

    df = log_json_to_df(data)

    df_normal = filter_normal_cases(df)

    pm4py_log = log_converter.apply(df_normal)

    dfg = dfg_discovery.apply(pm4py_log)

    dfg = prune_dfg(dfg, min_ratio=0.02)

    features_df = extract_case_features(df, dfg)

    features_df.to_csv("bpm/data/csv/dfg_case_features.csv", index=False)

    return df, dfg
