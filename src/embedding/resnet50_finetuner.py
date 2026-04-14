import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms


class ResNet50Finetuner:
    """
    Fine-tune ResNet50 pré-entraîné (ImageNet) sur CIFAR-10.
    - Remplace la tête FC par Linear(2048, 10)
    - Entraîne sur N images (défaut 1000)
    - Sauvegarde le state_dict dans data/models/resnet50_finetuned.pth
    """

    def __init__(self, device: str = None, batch_size: int = 64, num_epochs: int = 10, lr: float = 0.0001):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def finetune(self, root_dir: str = "./data/raw", n_train: int = 1000,
                 save_path: str = "./data/models/resnet50_finetuned.pth"):

        dataset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=self.transform)

        # Indices 0 -> n_train-1 réservés à l'entraînement
        train_subset = Subset(dataset, range(n_train))
        loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, 10)
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        print(f"Fine-tuning ResNet50 sur {n_train} images (train) | {self.num_epochs} epochs | {self.device}")

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0

            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"  Epoch {epoch+1}/{self.num_epochs} - Loss: {running_loss/len(loader):.4f}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Modèle sauvegardé -> {save_path}")

        return model