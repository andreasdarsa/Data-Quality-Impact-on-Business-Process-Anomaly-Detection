import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Correction: training only on the train set, core samples saved
def fit_dbscan_detector(
    X_train,
    eps=0.8,
    min_samples=5
):
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    model.fit(X_train)

    core_samples = model.components_

    if len(core_samples) == 0:
        raise ValueError(
            "DBSCAN did not identify any core samples. "
            "Consider adjusting eps or min_samples."
        )

    core_neighbors = NearestNeighbors(n_neighbors=1)
    core_neighbors.fit(core_samples)

    return {
        "model": model,
        "core_neighbors": core_neighbors,
        "eps": eps,
    }

# Anomaly score is not binary but is now based on the distance from the closest core sample
def evaluate_dbscan_detector(fitted_model, X_test):
    core_neighbors = fitted_model["core_neighbors"]
    eps = fitted_model["eps"]

    distances, _ = core_neighbors.kneighbors(X_test)

    test_scores = distances[:, 0]

    y_pred = (test_scores > eps).astype(int)

    return y_pred, test_scores
