import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]  
sys.path.append(str(ROOT))
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from src.knn.knn_search import KNNFinder
from src.dataset.cifar_loader import CIFARLoader


def tensor_to_hwc_numpy(img_tensor):
    # (C,H,W) -> (H,W,C)
    return img_tensor.permute(1, 2, 0).numpy()


def main():
    # 1) Charger embeddings et labels
    embeddings = np.load("./data/processed/embeddings.npy")
    N = embeddings.shape[0]

    # 2) Charger CIFAR-10 (subset N images)
    loader = CIFARLoader(root_dir="./data/raw", train=True)
    dataset = loader.load()
    subset = Subset(dataset, range(N))

    # 3) Construire KNN
    knn = KNNFinder(metric="cosine")
    knn.fit(embeddings)

    # 4) Choisir une image (id)
    idx = 100
    k = 10

    neighbors, distances = knn.query(idx, k=k)

    # 5) Affichage console
    img0, lbl0 = subset[idx]
    print(f"Image choisie: id={idx}, label={lbl0} ({dataset.classes[lbl0]})")

    print("\nTop 10 voisins:")
    for i, (nid, dist) in enumerate(zip(neighbors, distances), start=1):
        _, lbl = subset[nid]
        print(f"{i:02d}) id={nid}  label={lbl} ({dataset.classes[lbl]})  dist={dist:.4f}")


    cols = 5
    rows = 3 

    plt.figure(figsize=(cols * 3, rows * 3))

    # image sélectionnée
    plt.subplot(rows, cols, 1)
    plt.imshow(tensor_to_hwc_numpy(img0))
    plt.title(f"Selected\nid={idx}\n{dataset.classes[lbl0]}")
    plt.axis("off")

    # voisins
    for i, (nid, dist) in enumerate(zip(neighbors, distances), start=2):
        img, lbl = subset[nid]

        plt.subplot(rows, cols, i)
        plt.imshow(tensor_to_hwc_numpy(img))
        plt.title(f"id={nid}\n{dataset.classes[lbl]}\n{dist:.3f}")
        plt.axis("off")

    plt.suptitle("KNN - 10 plus proches voisins (Clip embeddings)", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
