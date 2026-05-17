import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def compute_metrics(
    embeddings: np.ndarray,
    coords_2d: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> dict:
    """
    Calcule 4 métriques standard pour évaluer la qualité des embeddings
    et de leur projection 2D.

    Args:
        embeddings: Vecteurs haute dimension (N, D).
        coords_2d:  Projection 2D (N, 2).
        labels:     Labels entiers (N,).
        k:          Nombre de voisins pour Trustworthiness et k-NN Accuracy.

    Returns:
        Dictionnaire avec les clés :
        - silhouette      : [-1, 1], plus haut = meilleur
        - davies_bouldin  : [0, ∞[, plus bas = meilleur
        - trustworthiness : [0, 1], plus haut = meilleur
        - knn_accuracy    : [0, 1], plus haut = meilleur
    """
    sil_score = silhouette_score(coords_2d, labels)

    db_score = davies_bouldin_score(coords_2d, labels)

    # Mesure si les voisins proches en haute dimension le restent en 2D
    trust_score = trustworthiness(embeddings, coords_2d, n_neighbors=k)

    # Accuracy k-NN en haute dimension (5-fold cross-val)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn_acc = cross_val_score(knn, embeddings, labels, cv=5, scoring="accuracy").mean()

    return {
        "silhouette":      float(sil_score),
        "davies_bouldin":  float(db_score),
        "trustworthiness": float(trust_score),
        "knn_accuracy":    float(knn_acc),
    }


def print_metrics(metrics: dict, k: int = 5, title: str = "MÉTRIQUES") -> None:
    """Affiche les métriques de façon formatée."""
    print(f"\n===== {title} =====")
    print(f"Silhouette Score      : {metrics['silhouette']:.4f}")
    print(f"Davies-Bouldin Index  : {metrics['davies_bouldin']:.4f}")
    print(f"Trustworthiness       : {metrics['trustworthiness']:.4f}")
    print(f"k-NN Accuracy (k={k}) : {metrics['knn_accuracy']:.4f}")


def save_metrics(metrics: dict, path: str, title: str, n: int, k: int) -> None:
    """Sauvegarde les métriques dans un fichier texte."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"===== {title} =====\n")
        f.write(f"Nombre d'images : {n}\n")
        f.write(f"k               : {k}\n\n")
        f.write(f"Silhouette Score     : {metrics['silhouette']:.4f}\n")
        f.write(f"Davies-Bouldin Index : {metrics['davies_bouldin']:.4f}\n")
        f.write(f"Trustworthiness      : {metrics['trustworthiness']:.4f}\n")
        f.write(f"k-NN Accuracy        : {metrics['knn_accuracy']:.4f}\n")
