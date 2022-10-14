import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import os

from .cifar10 import DictClassificationDataLoader


class ImageNet(pl.LightningDataModule):
    def __init__(self, n_workers: int, batch_size: int, extend_train: int = 1):
        super().__init__()
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.extend_train = extend_train
        self.n_classes = 1000
        self.root = os.environ.get("STTHREE_DATAPATH", "./data")

    def train_dataloader(self):
        trainset = ImageFolder(
            self.root,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
            ])
        )
        train_loader = DictClassificationDataLoader(
            trainset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_workers, pin_memory=True, extend=self.extend_train
        )
        return train_loader

    def val_dataloader(self):
        valset = ImageFolder(
            self.root,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        )
        val_loader = DictClassificationDataLoader(
            valset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_workers, pin_memory=True
        )
        return val_loader

    def test_dataloader(self):
        valset = ImageFolder(
            self.root,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        )
        val_loader = DictClassificationDataLoader(
            valset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_workers, pin_memory=True
        )
        return val_loader


