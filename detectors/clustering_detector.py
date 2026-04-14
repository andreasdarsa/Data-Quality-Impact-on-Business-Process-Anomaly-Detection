from sklearn.cluster import DBSCAN 
from detectors.base_detector import BaseDetector   


class DBSCANDetector(BaseDetector):
    def __init__(self, eps=0.5, min_samples=5):
        self.labels_ = None
        self.model = DBSCAN(eps=eps, min_samples=min_samples)

    def fit(self, X):
        self.labels_ = self.model.fit_predict(X)

    def predict(self, X):
        raise NotImplementedError("DBSCAN does not support predict on unseen data")

    def fit_predict(self, X):
        labels = self.model.fit_predict(X)
        return labels  # -1 anomaly >=0 clusters
