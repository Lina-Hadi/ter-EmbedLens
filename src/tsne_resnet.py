import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from dataset.cifar_loader import CIFARLoader
from embedding.resnet_extractor import ResNetExtractor
from reduction.tsne_reducer import TSNEReducer


N = 1000
K_NEIGHBORS = 5


def compute_metrics(embeddings, coords_2d, labels, k=5):
    """
    Calcule 4 métriques importantes pour évaluer les embeddings et la projection 2D.
    """

    # 1) Silhouette Score sur la projection 2D
    sil_score = silhouette_score(coords_2d, labels)

    # 2) Davies-Bouldin Index sur la projection 2D
    db_score = davies_bouldin_score(coords_2d, labels)

    # 3) Trustworthiness : respect du voisinage local entre 2048D et 2D
    trust_score = trustworthiness(embeddings, coords_2d, n_neighbors=k)

    # 4) k-NN Accuracy sur les embeddings originaux (2048D)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn_acc = cross_val_score(knn, embeddings, labels, cv=5, scoring="accuracy").mean()

    return {
        "silhouette": sil_score,
        "davies_bouldin": db_score,
        "trustworthiness": trust_score,
        "knn_accuracy": knn_acc,
    }


def main():
    # 1) Charger CIFAR-10
    print("Chargement CIFAR-10...")
    loader = CIFARLoader(root_dir="./data/raw", train=True)
    dataset = loader.load()
    subset = Subset(dataset, range(N))

    # 2) Extraire les embeddings ResNet50
    print(f"Extraction des embeddings ResNet50 sur {N} images...")
    extractor = ResNetExtractor(batch_size=64)
    embeddings, labels = extractor.extract(subset)
    print(f"Embeddings shape: {embeddings.shape}")  # (1000, 2048)

    # 3) Sauvegarder les embeddings et labels
    os.makedirs("./data/processed", exist_ok=True)
    np.save("./data/processed/embeddings_resnet_tsne.npy", embeddings)
    np.save("./data/processed/labels_resnet_tsne.npy", labels)
    print("Embeddings sauvegardés dans ./data/processed/")

    # 4) Réduction t-SNE
    print("Réduction t-SNE en 2D...")
    reducer = TSNEReducer(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=42
    )
    coords_2d = reducer.reduce(embeddings)
    np.save("./data/processed/coords_2d_resnet_tsne.npy", coords_2d)
    print("Coordonnées 2D sauvegardées.")

    # 5) Calcul des métriques
    print("Calcul des métriques...")
    metrics = compute_metrics(embeddings, coords_2d, labels, k=K_NEIGHBORS)

    print("\n===== MÉTRIQUES =====")
    print(f"Silhouette Score      : {metrics['silhouette']:.4f}")
    print(f"Davies-Bouldin Index  : {metrics['davies_bouldin']:.4f}")
    print(f"Trustworthiness       : {metrics['trustworthiness']:.4f}")
    print(f"k-NN Accuracy (k={K_NEIGHBORS}) : {metrics['knn_accuracy']:.4f}")

    # Sauvegarder les métriques dans un fichier texte
    with open("./data/processed/resnet_tsne_metrics.txt", "w", encoding="utf-8") as f:
        f.write("===== METRIQUES ResNet50 + t-SNE =====\n")
        f.write(f"Nombre d'images: {N}\n")
        f.write(f"k pour k-NN et Trustworthiness: {K_NEIGHBORS}\n\n")
        f.write(f"Silhouette Score     : {metrics['silhouette']:.4f}\n")
        f.write(f"Davies-Bouldin Index : {metrics['davies_bouldin']:.4f}\n")
        f.write(f"Trustworthiness      : {metrics['trustworthiness']:.4f}\n")
        f.write(f"k-NN Accuracy        : {metrics['knn_accuracy']:.4f}\n")

    # 6) Visualisation
    class_names = dataset.classes
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=8,
        alpha=0.7
    )

    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.ax.set_yticklabels(class_names)

    plt.title("CIFAR-10 - t-SNE 2D (ResNet50 embeddings)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.subplots_adjust(right=0.75)

    metrics_text = (
        f"Silhouette       : {metrics['silhouette']:.4f}\n"
        f"Davies-Bouldin   : {metrics['davies_bouldin']:.4f}\n"
        f"Trustworthiness  : {metrics['trustworthiness']:.4f}\n"
        f"k-NN Accuracy    : {metrics['knn_accuracy']:.4f}\n"
        f"k                : {K_NEIGHBORS}"
    )

    plt.figtext(
        0.78, 0.5,
        metrics_text,
        fontsize=10,
        va='center',
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black")
    )

    plt.savefig("./data/processed/resnet_tsne_plot_with_metrics.png", dpi=150)
    plt.show()

    print("Plot sauvegardé dans ./data/processed/resnet_tsne_plot_with_metrics.png")
    print("Métriques sauvegardées dans ./data/processed/resnet_tsne_metrics.txt")


if __name__ == "__main__":
    main()