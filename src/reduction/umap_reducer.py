import joblib
import umap


class UMAPReducer:
    """
    Wrapper autour de umap.UMAP.

    Améliorations par rapport à la version initiale :
    - Conserve le reducer fitté en attribut (self._reducer)
    - Peut le sauvegarder sur disque via joblib (save_path)
    - Expose transform() pour projeter de nouveaux points sans refaire le fit
      → indispensable pour Feature 4 (upload d'image → position dans le scatter)
    """

    def __init__(
        self,
        n_components: int = 2,
        random_state: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ):
        self.n_components  = n_components
        self.random_state  = random_state
        self.n_neighbors   = n_neighbors
        self.min_dist      = min_dist
        self._reducer      = None  # sera peuplé après reduce()

    # ------------------------------------------------------------------

    def reduce(self, embeddings, save_path: str = None):
        """
        Fit + transform : projette les embeddings en 2D.

        Args:
            embeddings: np.ndarray (N, D)
            save_path:  Si fourni, sauvegarde le reducer fitté avec joblib.
                        Exemple : "./data/processed/umap_reducer_resnet50_ft.joblib"

        Returns:
            coords_2d: np.ndarray (N, 2)
        """
        self._reducer = umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
        )
        coords_2d = self._reducer.fit_transform(embeddings)

        if save_path:
            joblib.dump(self._reducer, save_path)
            print(f"Reducer UMAP sauvegardé → {save_path}")

        return coords_2d

    # ------------------------------------------------------------------

    def transform(self, new_embeddings):
        """
        Projette de nouveaux embeddings dans l'espace 2D déjà appris.
        Nécessite que reduce() ait été appelé (ou que le reducer soit chargé).

        Args:
            new_embeddings: np.ndarray (M, D)

        Returns:
            coords_2d: np.ndarray (M, 2)
        """
        if self._reducer is None:
            raise RuntimeError(
                "Le reducer n'est pas fitté. "
                "Appelez reduce() d'abord, ou chargez un reducer avec load()."
            )
        return self._reducer.transform(new_embeddings)

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "UMAPReducer":
        """
        Charge un reducer préalablement sauvegardé et retourne
        une instance prête à utiliser transform().

        Usage :
            reducer = UMAPReducer.load("./data/processed/umap_reducer.joblib")
            new_coords = reducer.transform(new_embedding)
        """
        instance = cls()
        instance._reducer = joblib.load(path)
        return instance
