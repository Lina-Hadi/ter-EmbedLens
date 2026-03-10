import numpy as np
import umap


class UMAPReducer:
    def __init__(self, n_components=2, random_state=42, n_neighbors=15, min_dist=0.1):
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def reduce(self, embeddings):
        reducer = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist
        )

        coords_2d = reducer.fit_transform(embeddings)
        return coords_2d