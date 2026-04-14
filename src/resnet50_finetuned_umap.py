import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from dataset.cifar_loader import CIFARLoader
from embedding.resnet50_finetuner import ResNet50Finetuner
from embedding.resnet50_finetuned_extractor import ResNet50FinetunedExtractor
from reduction.umap_reducer import UMAPReducer


N_TRAIN = 1000   # images utilisées pour le fine-tuning (indices 0->999)
N_TEST  = 1000   # images utilisées pour l'évaluation (indices 1000->1999)
K_NEIGHBORS = 5
MODEL_PATH = "./data/models/resnet50_finetuned.pth"


def compute_metrics(embeddings, coords_2d, labels, k=5):
    sil_score = silhouette_score(coords_2d, labels)
    db_score = davies_bouldin_score(coords_2d, labels)
    trust_score = trustworthiness(embeddings, coords_2d, n_neighbors=k)
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

    # Séparation train / test (sans chevauchement)
    test_subset = Subset(dataset, range(N_TRAIN, N_TRAIN + N_TEST))

    # 2) Fine-tuning ResNet50 (si modèle absent) — sur les 1000 images TRAIN
    if not os.path.exists(MODEL_PATH):
        print("Modèle fine-tuné introuvable, lancement du fine-tuning...")
        finetuner = ResNet50Finetuner(batch_size=64, num_epochs=10, lr=0.0001)
        finetuner.finetune(root_dir="./data/raw", n_train=N_TRAIN, save_path=MODEL_PATH)
    else:
        print(f"Modèle fine-tuné trouvé -> {MODEL_PATH}")

    # 3) Extraire les embeddings — sur les 1000 images TEST (jamais vues pendant l'entraînement)
    print(f"Extraction des embeddings ResNet50 fine-tunés sur {N_TEST} images (test)...")
    extractor = ResNet50FinetunedExtractor(model_path=MODEL_PATH, batch_size=64)
    embeddings, labels = extractor.extract(test_subset)
    print(f"Embeddings shape: {embeddings.shape}")

    # 4) Sauvegarder
    os.makedirs("./data/processed", exist_ok=True)
    np.save("./data/processed/embeddings_resnet50_ft_umap.npy", embeddings)
    np.save("./data/processed/labels_resnet50_ft_umap.npy", labels)

    # 5) Réduction UMAP
    print("Réduction UMAP en 2D...")
    reducer = UMAPReducer(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords_2d = reducer.reduce(embeddings)
    np.save("./data/processed/coords_2d_resnet50_ft_umap.npy", coords_2d)

    # 6) Métriques
    print("Calcul des métriques...")
    metrics = compute_metrics(embeddings, coords_2d, labels, k=K_NEIGHBORS)

    print("\n===== MÉTRIQUES =====")
    print(f"Silhouette Score      : {metrics['silhouette']:.4f}")
    print(f"Davies-Bouldin Index  : {metrics['davies_bouldin']:.4f}")
    print(f"Trustworthiness       : {metrics['trustworthiness']:.4f}")
    print(f"k-NN Accuracy (k={K_NEIGHBORS}) : {metrics['knn_accuracy']:.4f}")

    with open("./data/processed/resnet50_ft_umap_metrics.txt", "w", encoding="utf-8") as f:
        f.write("===== METRIQUES ResNet50 Fine-tuné + UMAP =====\n")
        f.write(f"Nombre d'images: {N_TRAIN}\n")
        f.write(f"k pour k-NN et Trustworthiness: {K_NEIGHBORS}\n\n")
        f.write(f"Silhouette Score     : {metrics['silhouette']:.4f}\n")
        f.write(f"Davies-Bouldin Index : {metrics['davies_bouldin']:.4f}\n")
        f.write(f"Trustworthiness      : {metrics['trustworthiness']:.4f}\n")
        f.write(f"k-NN Accuracy        : {metrics['knn_accuracy']:.4f}\n")

    # 7) Visualisation
    class_names = dataset.classes
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.ax.set_yticklabels(class_names)

    plt.title("CIFAR-10 - UMAP 2D (ResNet50 Fine-tuné embeddings)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.subplots_adjust(right=0.75)

    metrics_text = (
        f"Silhouette       : {metrics['silhouette']:.4f}\n"
        f"Davies-Bouldin   : {metrics['davies_bouldin']:.4f}\n"
        f"Trustworthiness  : {metrics['trustworthiness']:.4f}\n"
        f"k-NN Accuracy    : {metrics['knn_accuracy']:.4f}\n"
        f"k                : {K_NEIGHBORS}"
    )
    plt.figtext(0.78, 0.5, metrics_text, fontsize=10, va='center',
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="black"))

    plt.savefig("./data/processed/resnet50_ft_umap_plot.png", dpi=150)
    plt.show()
    print("Plot sauvegardé -> ./data/processed/resnet50_ft_umap_plot.png")


if __name__ == "__main__":
    main()