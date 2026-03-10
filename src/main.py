import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from dataset.cifar_loader import CIFARLoader
from embedding.extractor import EmbeddingExtractor
from reduction.umap_reducer import UMAPReducer


N = 10000  # nombre d'images à traiter (max 50000 pour train)


def main():
    # 1) Charger CIFAR-10
    print("Chargement CIFAR-10...")
    loader = CIFARLoader(root_dir="./data/raw", train=True)
    dataset = loader.load()
    subset = Subset(dataset, range(N))

    # 2) Extraire les embeddings CLIP
    print(f"Extraction des embeddings CLIP sur {N} images...")
    extractor = EmbeddingExtractor(batch_size=64)
    embeddings, labels = extractor.extract(subset)
    print(f"Embeddings shape: {embeddings.shape}")  # (N, 512)

    # 3) Sauvegarder les embeddings et labels
    import os
    os.makedirs("./data/processed", exist_ok=True)
    np.save("./data/processed/embeddings.npy", embeddings)
    np.save("./data/processed/labels.npy", labels)
    print("Embeddings sauvegardés dans ./data/processed/")

    # 4) Réduction UMAP
    print("Réduction UMAP en 2D...")
    reducer = UMAPReducer(n_components=2, n_neighbors=15, min_dist=0.1)
    coords_2d = reducer.reduce(embeddings)
    np.save("./data/processed/coords_2d.npy", coords_2d)
    print("Coordonnées 2D sauvegardées.")

    # 5) Visualisation
    class_names = dataset.classes
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=labels, cmap="tab10", s=8, alpha=0.7
    )
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.ax.set_yticklabels(class_names)
    plt.title("CIFAR-10 - UMAP 2D (CLIP embeddings)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig("./data/processed/umap_plot.png", dpi=150)
    plt.show()
    print("Plot sauvegardé dans ./data/processed/umap_plot.png")


if __name__ == "__main__":
    main()