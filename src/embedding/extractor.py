import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image
import clip


class CLIPDatasetWrapper(Dataset):
    """Applique le préprocesseur CLIP à chaque image du dataset."""
    def __init__(self, subset, preprocess):
        self.subset = subset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # image est un tensor (C,H,W) entre 0 et 1 -> convertir en PIL pour CLIP
        image_pil = to_pil_image(image)
        image_clip = self.preprocess(image_pil)
        return image_clip, label


class EmbeddingExtractor:

    def __init__(self, device: str = None, batch_size: int = 64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Modèle CLIP (ViT-B/32) pré-entraîné
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def extract(self, dataset_subset):
        # Wrapper pour appliquer le bon préprocesseur CLIP (resize 224x224, normalize, etc.)
        wrapped = CLIPDatasetWrapper(dataset_subset, self.preprocess)
        loader = DataLoader(wrapped, batch_size=self.batch_size, shuffle=False)

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)

                # CLIP encode_image retourne (batch_size, 512)
                emb = self.model.encode_image(images)

                # Normalisation L2 (recommandée avec CLIP)
                emb = emb / emb.norm(dim=-1, keepdim=True)

                all_embeddings.append(emb.cpu().float().numpy())
                all_labels.append(labels.numpy())

        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)

        return embeddings, labels