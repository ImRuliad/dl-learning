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


if __name__ == "__main__":
    transform = transforms.ToTensor()
    create_train_dataset(transform)
    #create_test_dataset(transform)
