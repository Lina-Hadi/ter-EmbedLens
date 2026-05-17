import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, datasets, transforms


class ResNet50Finetuner:
    """
    Fine-tune ResNet50 pré-entraîné (ImageNet) sur CIFAR-10.

    Améliorations par rapport à la version initiale :
    - Split train / validation explicite (80/20 sur le sous-ensemble)
    - Sauvegarde uniquement le meilleur modèle (selon val accuracy)
    - Learning rate scheduler (CosineAnnealingLR)
    - Historique d'entraînement sauvegardé en JSON
    - Affichage de la val accuracy à chaque epoch
    """

    def __init__(
        self,
        device: str = None,
        batch_size: int = 64,
        num_epochs: int = 10,
        lr: float = 0.0001,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        # Transforms ImageNet standard — identique à l'extractor pour cohérence
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    # ------------------------------------------------------------------
    # Méthode principale
    # ------------------------------------------------------------------

    def finetune(
        self,
        root_dir: str = "./data/raw",
        n_train= None,
        save_path: str = "./data/models/resnet50_finetuned.pth",
    ):
        """
        Lance le fine-tuning et sauvegarde le meilleur checkpoint.

        Args:
            root_dir:   Dossier racine de CIFAR-10.
            n_train:    Nombre d'images utilisées pour entraîner (indices 0 → n_train-1).
                        80 % seront utilisées pour le train, 20 % pour la validation.
            save_path:  Chemin du fichier .pth pour le meilleur modèle.

        Returns:
            Le modèle PyTorch avec les meilleurs poids chargés.
        """
        # 1) Données
        # Entraînement sur les 50 000 images du train set officiel
        train_dataset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=self.transform)

        # Évaluation sur les 10 000 images du test set officiel (jamais vues pendant l'entraînement)
        test_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=self.transform)

        val_size   = int(len(train_dataset) * 0.2)  # 10 000 images pour la validation
        train_size = len(train_dataset) - val_size   # 40 000 images pour l'entraînement

        train_subset = Subset(train_dataset, range(train_size))
        val_subset   = Subset(train_dataset, range(train_size, len(train_dataset)))

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

        print(
            f"\nFine-tuning ResNet50 | train={train_size} | val={val_size} | "
            f"epochs={self.num_epochs} | device={self.device}"
        )

        # 2) Modèle
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, 10)
        model = model.to(self.device)

        # 3) Optimiseur + scheduler + loss
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # CosineAnnealingLR : réduit le LR progressivement jusqu'à 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )

        # 4) Boucle d'entraînement
        history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
        best_val_acc = 0.0
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(self.num_epochs):
            # --- Phase train ---
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # --- Phase validation ---
            val_loss, val_acc = self._evaluate(model, val_loader, criterion)

            # --- Scheduler step ---
            scheduler.step()

            # --- Log ---
            print(
                f"  Epoch {epoch + 1:02d}/{self.num_epochs} "
                f"| train_loss={avg_train_loss:.4f} "
                f"| val_loss={val_loss:.4f} "
                f"| val_acc={val_acc * 100:.1f}%"
                + (" ← best" if val_acc > best_val_acc else "")
            )

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(round(avg_train_loss, 6))
            history["val_loss"].append(round(val_loss, 6))
            history["val_accuracy"].append(round(val_acc, 6))

            # --- Sauvegarder uniquement si meilleur ---
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)

        print(f"\nMeilleur modèle sauvegardé → {save_path} (val_acc={best_val_acc * 100:.1f}%)")

        # 5) Sauvegarder l'historique JSON à côté du .pth
        history_path = save_path.replace(".pth", "_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Historique sauvegardé → {history_path}")

        # 6) Recharger les meilleurs poids avant de retourner
        model.load_state_dict(torch.load(save_path, map_location=self.device))
        return model

    # ------------------------------------------------------------------
    # Méthode privée d'évaluation
    # ------------------------------------------------------------------

    def _evaluate(self, model, loader, criterion):
        """Calcule la loss et l'accuracy sur un DataLoader."""
        model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                total_loss += loss.item()
                predicted   = outputs.argmax(dim=1)
                correct    += (predicted == labels).sum().item()
                total      += labels.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy
