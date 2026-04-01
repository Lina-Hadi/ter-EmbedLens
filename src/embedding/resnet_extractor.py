import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms


class ResNetExtractor:
    """
    Extrait les embeddings ResNet50 pré-entraîné (ImageNet).
    La dernière couche FC est retirée -> embeddings de dim 2048.
    """

    def __init__(self, device: str = None, batch_size: int = 64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # ResNet50 pré-entraîné, on retire la tête de classification
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(backbone.children())[:-1])  # -> (B, 2048, 1, 1)
        self.model.to(self.device)
        self.model.eval()

        # Préprocessing standard ImageNet
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, dataset_subset):
        loader = DataLoader(dataset_subset, batch_size=self.batch_size, shuffle=False)

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)

                # Normalisation ImageNet (les images sont déjà des tensors 0-1 via CIFARLoader)
                images = self.preprocess(images)

                # (B, 2048, 1, 1) -> (B, 2048)
                emb = self.model(images).squeeze(-1).squeeze(-1)

                # Normalisation L2
                emb = emb / emb.norm(dim=-1, keepdim=True)

                all_embeddings.append(emb.cpu().float().numpy())
                all_labels.append(labels.numpy())

        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)

        return embeddings, labels