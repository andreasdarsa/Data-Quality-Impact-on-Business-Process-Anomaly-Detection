import numpy as np
from sklearn.neighbors import NearestNeighbors

# Keep the same logic: distance from 5th nn, 90th percentile threshold
def fit_knn_detector(
    X_train,
    threshold_percentile=90,
    n_neighbors=5
):
    model = NearestNeighbors(
        n_neighbors=n_neighbors
    )

    model.fit(X_train)

    # Exclude each training sample from its own neighbors.
    train_distances, _ = model.kneighbors(
        X_train,
        n_neighbors=n_neighbors + 1
    )

    train_scores = train_distances[:, -1]

    threshold = np.percentile(
        train_scores,
        threshold_percentile
    )

    return {
        "model": model,
        "threshold": threshold,
        "n_neighbors": n_neighbors,
    }


def evaluate_knn_detector(fitted_model, X_test):
    model = fitted_model["model"]
    threshold = fitted_model["threshold"]
    n_neighbors = fitted_model["n_neighbors"]

    test_distances, _ = model.kneighbors(
        X_test,
        n_neighbors=n_neighbors
    )

    test_scores = test_distances[:, -1]

    y_pred = (test_scores > threshold).astype(int)

    return y_pred, test_scores
