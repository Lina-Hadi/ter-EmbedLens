import os
from torchvision import datasets, transforms


class CIFARLoader:
    def __init__(self, root_dir: str = "./data/raw", train: bool = True, image_size: int = 224):
        self.root_dir = root_dir
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # créer le dossier si besoin
        os.makedirs(self.root_dir, exist_ok=True)

    def load(self):
        """
        Télécharge (si besoin) et retourne le dataset CIFAR-10.
        """
        dataset = datasets.CIFAR10(
            root=self.root_dir,
            train=self.train,
            download=True,
            transform=self.transform
        )
        return dataset