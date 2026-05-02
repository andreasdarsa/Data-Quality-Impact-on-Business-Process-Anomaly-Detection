import numpy as np
from sklearn.neighbors import NearestNeighbors


def run_knn_detector(X_train, X_test, threshold_percentile=90, n_neighbors=5):
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(X_train)

    train_distances, _ = model.kneighbors(X_train)
    test_distances, _ = model.kneighbors(X_test)

    train_scores = train_distances[:, -1]
    test_scores = test_distances[:, -1]

    threshold = np.percentile(train_scores, threshold_percentile)

    y_pred = (test_scores > threshold).astype(int)

    return y_pred, test_scores
