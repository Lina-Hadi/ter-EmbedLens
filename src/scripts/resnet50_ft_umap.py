"""
Pipeline : ResNet50 Fine-tuné → UMAP

Workflow complet :
  1. Charge CIFAR-10
  2. Fine-tune ResNet50 sur les indices [0, N_TRAIN[ (si checkpoint absent)
  3. Extrait les embeddings sur les indices [N_TRAIN, N_TRAIN + N_TEST[
  4. Réduit en 2D avec UMAP et sauvegarde le reducer fitté (.joblib)
  5. Calcule et sauvegarde les métriques
  6. Produit et sauvegarde le plot matplotlib

Usage :
  python src/scripts/resnet50_ft_umap.py
  python src/scripts/resnet50_ft_umap.py --n_train 2000 --n_test 1000 --epochs 15
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

# Permettre les imports absolus depuis la racine du projet
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.dataset.cifar_loader import CIFARLoader
from src.training.finetuner import ResNet50Finetuner
from src.embedding.resnet50_finetuned_extractor import ResNet50FinetunedExtractor
from src.reduction.umap_reducer import UMAPReducer
from src.utils.metrics import compute_metrics, print_metrics, save_metrics


# ------------------------------------------------------------------
# Constantes par défaut (surchargeables via argparse)
# ------------------------------------------------------------------
DEFAULT_N_TEST     = 10000
DEFAULT_K          = 5
DEFAULT_EPOCHS     = 10
DEFAULT_LR         = 0.0001
DEFAULT_MODEL_PATH = "./data/models/resnet50_finetuned.pth"
PROCESSED_DIR      = "./data/processed"


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def plot_and_save(coords_2d, labels, class_names, metrics, k, save_path):
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=labels, cmap="tab10", s=8, alpha=0.7,
    )
    cbar = plt.colorbar(scatter, ticks=range(10))
    cbar.ax.set_yticklabels(class_names)

    plt.title("CIFAR-10 — UMAP 2D (ResNet50 Fine-tuné)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.subplots_adjust(right=0.75)

    metrics_text = (
        f"Silhouette       : {metrics['silhouette']:.4f}\n"
        f"Davies-Bouldin   : {metrics['davies_bouldin']:.4f}\n"
        f"Trustworthiness  : {metrics['trustworthiness']:.4f}\n"
        f"k-NN Accuracy    : {metrics['knn_accuracy']:.4f}\n"
        f"k                : {k}"
    )
    plt.figtext(
        0.78, 0.5, metrics_text, fontsize=10, va="center",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black"),
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot sauvegardé → {save_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1) Charger CIFAR-10
    print("Chargement CIFAR-10...")
    loader  = CIFARLoader(root_dir="./data/raw", train=True)
    dataset = loader.load()

    # Séparation stricte train / test (aucun chevauchement)
    test_loader = CIFARLoader(root_dir="./data/raw", train=False)
    test_dataset = test_loader.load()
    test_subset = Subset(test_dataset, range(args.n_test))

    # 2) Fine-tuning (ignoré si le checkpoint existe déjà)
    if not os.path.exists(args.model_path):
        print(f"\nCheckpoint absent → lancement du fine-tuning ({args.epochs} epochs)...")
        finetuner = ResNet50Finetuner(
            batch_size=64,
            num_epochs=args.epochs,
            lr=args.lr,
        )
        finetuner.finetune(
            root_dir="./data/raw",
            save_path=args.model_path,
        )
    else:
        print(f"\nCheckpoint trouvé → {args.model_path} (fine-tuning ignoré)")

    # 3) Extraction des embeddings sur le jeu de TEST
    print(f"\nExtraction des embeddings ResNet50 fine-tuné sur {args.n_test} images (test)...")
    extractor = ResNet50FinetunedExtractor(
        model_path=args.model_path, batch_size=64
    )
    embeddings, labels = extractor.extract(test_subset)
    print(f"Embeddings shape : {embeddings.shape}")  # (N_TEST, 2048)

    # 4) Sauvegarde des embeddings et labels
    emb_path    = os.path.join(PROCESSED_DIR, "embeddings_resnet50_ft_umap.npy")
    labels_path = os.path.join(PROCESSED_DIR, "labels_resnet50_ft_umap.npy")
    np.save(emb_path,    embeddings)
    np.save(labels_path, labels)
    print(f"Embeddings sauvegardés → {emb_path}")

    # 5) Réduction UMAP
    # Le reducer fitté est sauvegardé pour pouvoir projeter de nouveaux points
    # plus tard (Feature 4 — upload d'image).
    print("\nRéduction UMAP en 2D...")
    reducer       = UMAPReducer(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    reducer_path  = os.path.join(PROCESSED_DIR, "umap_reducer_resnet50_ft.joblib")
    coords_2d     = reducer.reduce(embeddings, save_path=reducer_path)

    coords_path = os.path.join(PROCESSED_DIR, "coords_2d_resnet50_ft_umap.npy")
    np.save(coords_path, coords_2d)
    print(f"Coordonnées 2D sauvegardées → {coords_path}")

    # 6) Métriques
    print("\nCalcul des métriques...")
    metrics = compute_metrics(embeddings, coords_2d, labels, k=args.k)
    print_metrics(metrics, k=args.k, title="ResNet50 Fine-tuné + UMAP")

    metrics_path = os.path.join(PROCESSED_DIR, "resnet50_ft_umap_metrics.txt")
    save_metrics(
        metrics,
        path=metrics_path,
        title="ResNet50 Fine-tuné + UMAP",
        n=args.n_test,
        k=args.k,
    )
    print(f"Métriques sauvegardées → {metrics_path}")

    # 7) Visualisation
    plot_path = os.path.join(PROCESSED_DIR, "resnet50_ft_umap_plot.png")
    plot_and_save(coords_2d, labels, dataset.classes, metrics, args.k, plot_path)


# ------------------------------------------------------------------
# Entrée CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline ResNet50 Fine-tuné + UMAP sur CIFAR-10"
    )
    parser.add_argument(
        "--n_test", type=int, default=DEFAULT_N_TEST,
        help=f"Nombre d'images pour l'évaluation (défaut : {DEFAULT_N_TEST})",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Nombre d'epochs de fine-tuning (défaut : {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Learning rate (défaut : {DEFAULT_LR})",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Nombre de voisins pour les métriques (défaut : {DEFAULT_K})",
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Chemin du checkpoint .pth (défaut : {DEFAULT_MODEL_PATH})",
    )
    main(parser.parse_args())
