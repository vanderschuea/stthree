import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import torchvision.transforms as transforms
import os


class Cifar10(pl.LightningDataModule):
    def __init__(self, n_workers: int, max_perc: float, batch_size: int, extend_train: int = 1):
        super().__init__()
        self.n_workers = n_workers
        self.max_perc = max_perc
        self.batch_size = batch_size
        self.extend_train = extend_train
        self.n_classes = 10
        self.root = os.environ.get("STTHREE_DATAPATH", "./data")

    def prepare_data(self) -> None:
        torchvision.datasets.CIFAR10(root=self.root, train=True, download=True)

    def setup(self, stage) -> None:
        n_train = 50000
        self.indices = np.arange(n_train)
        self.val_split = int(0.1 * n_train)
        split = n_train - self.val_split
        self.train_split = round(self.max_perc*split)

    def train_dataloader(self) -> DataLoader:
        trainset = torchvision.datasets.CIFAR10(
            root=self.root, train=True, download=False, transform=transforms.Compose([
                transforms.RandomCrop(32, 4, padding_mode="edge"), transforms.ToTensor(),
            ])
        )
        trainset = Subset(trainset, self.indices[:self.train_split])
        train_loader = DictClassificationDataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, extend=self.extend_train, pin_memory=True,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        valset = torchvision.datasets.CIFAR10(
            root=self.root, train=True, download=False, transform=transforms.ToTensor()
        )
        valset = Subset(valset, self.indices[-self.val_split:])
        val_loader = DictClassificationDataLoader(
            valset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        testset = torchvision.datasets.CIFAR10(
            root=self.root, train=False, download=False, transform=transforms.ToTensor()
        )
        test_loader = DictClassificationDataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True
        )
        return test_loader


class DictClassificationDataLoader(DataLoader):
    # TODO: support memory pinning by returning custom type!!
    def __init__(self, *args, extend=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.extend = extend
    def __len__(self) -> int:
        return super().__len__()*self.extend
    def __iter__(self):
        for inputs, outputs  in super().__iter__():
            yield {
                "image": inputs,
                "classification": outputs
            }
            for _ in range(self.extend-1):
                yield {
                    "image": None,
                    "classification": None
                }
