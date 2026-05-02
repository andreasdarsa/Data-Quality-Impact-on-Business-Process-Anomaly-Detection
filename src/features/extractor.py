import pandas as pd

from .basic import extract_basic_features
from .temporal import extract_temporal_features
from .resource import extract_resource_features
from .control_flow import extract_control_flow_features


def extract_features(logs: list[dict], dfg: dict):
    rows = []
    labels = []
    subtypes = []

    for case in logs:
        features = {}

        features.update(extract_basic_features(case))
        features.update(extract_temporal_features(case))
        features.update(extract_resource_features(case))
        features.update(extract_control_flow_features(case, dfg))

        rows.append(features)
        labels.append(int(case.get("is_anomaly", False)))
        subtypes.append(case.get("subtype", "normal"))

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="is_anomaly")
    subtype = pd.Series(subtypes, name="subtype")

    return X, y, subtype
