import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms


class ResNet50FinetunedExtractor:
    """
    Extrait les embeddings depuis un ResNet50 fine-tuné sur CIFAR-10.
    La tête FC (Linear 2048->10) est remplacée par Identity() après chargement
    -> embeddings de dim 2048, normalisés L2.
    """

    def __init__(self, model_path: str = "./data/models/resnet50_finetuned.pth",
                 device: str = None, batch_size: int = 64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Charger le modèle fine-tuné
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(2048, 10)
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Retirer la tête -> embeddings 2048D
        model.fc = nn.Identity()
        model.to(self.device)
        model.eval()

        self.model = model

    def extract(self, dataset_subset):
        loader = DataLoader(dataset_subset, batch_size=self.batch_size, shuffle=False)

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                images = self.preprocess(images)

                emb = self.model(images)  # (B, 2048)
                emb = emb / emb.norm(dim=-1, keepdim=True)

                all_embeddings.append(emb.cpu().float().numpy())
                all_labels.append(labels.numpy())

        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)

        return embeddings, labels