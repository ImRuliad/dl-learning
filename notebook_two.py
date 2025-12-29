import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_train_dataset(transform):
    try:
        train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
        )
    except Exception as e:
        print(f"Unable to obtain train dataset {e}")
        return
    print(f"Successfully obtained train dataset of size {len(train_dataset)}")
    return train_dataset

def create_test_dataset(transform):
    try:
        test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
        )
    except Exception as e:
        print(f"Unable to obtain test dataset {e}")
        return
    print(f"Successfully obtained test dataset of size {len(test_dataset)}")
    return test_dataset

def create_dataloaders(dataset, batch_size, shuffle):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    print(f"Successfully created dataloader for {dataset} with BATCH SIZE {batch_size}")
    return dataloader


if __name__ == "__main__":
    transform = transforms.ToTensor()
    BATCH_SIZE = 64

    train_dataset = create_train_dataset(transform)
    test_dataset = create_test_dataset(transform)

    create_dataloaders(train_dataset, BATCH_SIZE, shuffle=True)
    create_dataloaders(test_dataset, BATCH_SIZE, shuffle=False)
