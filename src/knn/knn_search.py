import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNFinder:
    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.nn = None
        self.embeddings = None

    def fit(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        self.nn = NearestNeighbors(metric=self.metric)
        self.nn.fit(embeddings)

    def query(self, idx: int, k: int = 5):
        distances, indices = self.nn.kneighbors(
            self.embeddings[idx:idx+1], n_neighbors=k + 1, return_distance=True
        )
        indices = indices[0].tolist()
        distances = distances[0].tolist()

        if indices and indices[0] == idx:
            indices = indices[1:]
            distances = distances[1:]

        return indices, distances
