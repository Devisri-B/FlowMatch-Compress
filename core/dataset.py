"""
Dataset loaders and real-world image preprocessors for FlowMatch-Compress.
Supports:
1. CIFAR-10 Photographic Benchmark (50,000 real photos across 10 classes)
2. High-Resolution Latent Dataset (Photorealistic scenes encoded via VAE)
"""

import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T


CIFAR10_CLASSES = [
    "Airplane ✈️",
    "Automobile 🚗",
    "Bird 🐦",
    "Cat 🐱",
    "Deer 🦌",
    "Dog 🐶",
    "Frog 🐸",
    "Horse 🐴",
    "Ship 🚢",
    "Truck 🚚"
]


def get_cifar10_loader(
    data_dir: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Returns DataLoader for normalized CIFAR-10 real photographic images.
    Images normalized to [-1.0, 1.0].
    """
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
