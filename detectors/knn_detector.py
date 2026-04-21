from sklearn.neighbors import NearestNeighbors
from detectors.base_detector import BaseDetector
import numpy as np


class KNNDetector(BaseDetector):
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=n_neighbors)

    def fit(self, X):
        self.model.fit(X)

    def score_samples(self, X):
        distances, _ = self.model.kneighbors(X)
        return distances.mean(axis=1)  # anomaly score

    def predict(self, X, percentile=90):
        scores = self.score_samples(X)
        threshold = np.percentile(scores, percentile)
        return (scores >= threshold).astype(int)