"""Dataset and DataLoader construction for the BUS multi-task problem."""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from medimeta import MedIMeta

import config

import numpy as np
from collections import Counter

def compute_class_weights(train_ds):
    """Return (weights_a, pos_weight_b) computed from the training labels."""
    labels_a, labels_b = [], []
    for i in range(len(train_ds)):
        _, la, lb = train_ds[i]
        labels_a.append(int(la))
        labels_b.append(int(lb.item()))

    # Head A
    counts_a = Counter(labels_a)
    n_a = len(labels_a)
    weights_a = torch.tensor(
        [n_a / (len(counts_a) * counts_a.get(c, 1)) for c in range(3)],
        dtype=torch.float32,
    )

    # Head B
    n_pos = sum(labels_b)
    n_neg = len(labels_b) - n_pos
    pos_weight_b = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)

    return weights_a, pos_weight_b


class BUSMultiTaskDataset(Dataset):
    """
    Wraps two MedIMeta datasets (one per task) so that each __getitem__
    returns (image, label_a, label_b) for the *same* image index.
    """

    def __init__(
        self,
        data_root: str,
        dataset_id: str,
        task_a: str,
        task_b: str,
        split: str = "train",
        transform=None,
    ):
        self.ds_a = MedIMeta(data_root, dataset_id, task_a, split=split)
        self.ds_b = MedIMeta(data_root, dataset_id, task_b, split=split)

        assert len(self.ds_a) == len(self.ds_b), (
            "Task A and Task B datasets must have the same number of samples."
        )

        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.ds_a)

    def __getitem__(self, idx):
        img_a, label_a = self.ds_a[idx]
        img_b, label_b = self.ds_b[idx]

        if self.transform is not None:
            img = self.transform(img_a)
        else:
            img = img_a

        # --- Robust label handling ---
        label_a = torch.as_tensor(label_a).detach().clone().long().reshape(-1)[0]
        label_b = torch.as_tensor(label_b).detach().clone().float().reshape(-1)[0]

        return img, label_a, label_b


def build_transforms(train: bool = True):
    # MedIMeta returns float tensors in [0, 1].
    # ResNet18 expects ImageNet-normalized inputs.
    normalize = transforms.Normalize(
        mean=[0.449], std=[0.226],
    )

    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            normalize,
        ])
    return transforms.Compose([normalize])


def build_dataloaders():
    device = config.get_device()
    pin = device.type == "cuda"

    train_ds = BUSMultiTaskDataset(
        data_root=config.DATA_ROOT,
        dataset_id=config.DATASET_ID,
        task_a=config.TASK_A_NAME,
        task_b=config.TASK_B_NAME,
        split="train",
        transform=build_transforms(train=True),
    )
    val_ds = BUSMultiTaskDataset(
        data_root=config.DATA_ROOT,
        dataset_id=config.DATASET_ID,
        task_a=config.TASK_A_NAME,
        task_b=config.TASK_B_NAME,
        split="val",
        transform=build_transforms(train=False),
    )
    test_ds = BUSMultiTaskDataset(
        data_root=config.DATA_ROOT,
        dataset_id=config.DATASET_ID,
        task_a=config.TASK_A_NAME,
        task_b=config.TASK_B_NAME,
        split="test",
        transform=build_transforms(train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader