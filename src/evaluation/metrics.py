from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_detector_results(y_true, detector_results: dict) -> dict:
    results = {}

    for detector_name, output in detector_results.items():
        results[detector_name] = compute_metrics(
            y_true,
            output["y_pred"]
        )

    return results
