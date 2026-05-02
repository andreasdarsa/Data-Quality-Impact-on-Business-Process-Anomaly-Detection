from sklearn.preprocessing import StandardScaler

from .knn import run_knn_detector
from .lof import run_lof_detector
from .iforest import run_iforest_detector
from .dbscan import run_dbscan_detector


def run_all_detectors(
    X_train,
    X_test,
    threshold_percentile=90,
    seed=42
):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    y_pred, scores = run_knn_detector(
        X_train_scaled,
        X_test_scaled,
        threshold_percentile=threshold_percentile
    )
    results["knn"] = {
        "y_pred": y_pred,
        "scores": scores,
    }

    y_pred, scores = run_lof_detector(
        X_train_scaled,
        X_test_scaled,
        threshold_percentile=threshold_percentile
    )
    results["lof"] = {
        "y_pred": y_pred,
        "scores": scores,
    }

    y_pred, scores = run_iforest_detector(
        X_train_scaled,
        X_test_scaled,
        threshold_percentile=threshold_percentile,
        seed=seed
    )
    results["iforest"] = {
        "y_pred": y_pred,
        "scores": scores,
    }

    y_pred, scores = run_dbscan_detector(
        X_train_scaled,
        X_test_scaled
    )
    results["dbscan"] = {
        "y_pred": y_pred,
        "scores": scores,
    }

    return results
