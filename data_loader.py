import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from config import *


def get_transforms():
    """Return train and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, val_transform


def get_dataloaders():
    """Return train and validation dataloaders"""
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, train_dataset.classes