from detectors.base_detector import BaseDetector
from sklearn.neighbors import LocalOutlierFactor


class LOFDetector(BaseDetector):
    def __init__(self, n_neighbors=20):
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        preds = self.model.predict(X)
        return (preds == -1).astype(int)

    def score_samples(self, X):
        return -self.model.decision_function(X)  # higher = more anomalous
