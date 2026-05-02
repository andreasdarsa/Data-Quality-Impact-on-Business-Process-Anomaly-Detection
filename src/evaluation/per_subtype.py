import pandas as pd


def evaluate_per_subtype(y_true, subtype, detector_results: dict) -> dict:
    """
    Evaluates anomaly detection performance per anomaly subtype.

    Important:
    Per-subtype evaluation contains only anomalous cases of that subtype.
    Therefore precision/F1/confusion matrix are not meaningful here.

    We report:
    - detected: how many cases of this subtype were detected as anomalies
    - total: total cases of this subtype
    - recall: detected / total
    """

    df = pd.DataFrame({
        "y_true": y_true,
        "subtype": subtype,
    })

    df = df[df["subtype"] != "normal"]

    results = {}

    for detector_name, output in detector_results.items():
        detector_df = df.copy()
        detector_df["y_pred"] = output["y_pred"][detector_df.index]

        subtype_results = {}

        for subtype_name in sorted(detector_df["subtype"].unique()):
            subset = detector_df[detector_df["subtype"] == subtype_name]

            total = len(subset)
            detected = int(subset["y_pred"].sum())

            subtype_results[subtype_name] = {
                "detected": detected,
                "total": total,
                "recall": detected / total if total else 0.0,
            }

        results[detector_name] = subtype_results

    return results
