import numpy as np
from sklearn.manifold import TSNE


class TSNEReducer:
    def __init__(self, n_components=2, random_state=42, perplexity=30, learning_rate=200, max_iter=1000):
        self.n_components = n_components
        self.random_state = random_state
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def reduce(self, embeddings):
        reducer = TSNE(
            n_components=self.n_components,
            random_state=self.random_state,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter
        )

        coords_2d = reducer.fit_transform(embeddings)
        return coords_2d