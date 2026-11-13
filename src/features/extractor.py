
import numpy as np
import pandas as pd

from .basic import extract_basic_features
from .temporal import extract_temporal_features
from .resource import extract_resource_features
from .control_flow import extract_control_flow_features


def extract_features(
    logs: list[dict],
    dfg: dict,
    dataset: str = "bpic12",
):
    rows = []
    labels = []
    subtypes = []

    for case in logs:
        features = {}

        features.update(extract_basic_features(case))
        features.update(extract_temporal_features(case))
        features.update(extract_resource_features(case))
        features.update(extract_control_flow_features(case, dfg))

        # BPIC13 does not provide amount or event duration data.
        if dataset.lower() == "bpic13":
            features.pop("amount", None)

            for key in (
                "mean_duration",
                "std_duration",
                "min_duration",
                "max_duration",
            ):
                features.pop(key, None)

        elif dataset.lower() != "bpic12":
            raise ValueError(f"Unsupported dataset: {dataset}")

        rows.append(features)
        labels.append(int(case.get("is_anomaly", False)))
        subtypes.append(case.get("subtype", "normal"))

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="is_anomaly")
    subtype = pd.Series(subtypes, name="subtype")

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in X.dtypes
    ):
        raise ValueError("Feature matrix contains non-numeric columns.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    return X, y, subtype
