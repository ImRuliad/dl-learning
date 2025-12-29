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

def get_single_data_batch(dataloader):
    x, y = next(iter(dataloader))
    print("Image tensor shape:", x.shape)
    print("Label tensor shape:", y.shape)
    print(f"\nExpected image shape: (64, 1, 28, 28)")
    print(f"Expected label shape: (64,)")
    print(f"\nLabels in this batch: {y[:10].tolist()}...")

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()             #Flatten 28x28 image to 784 
        self.linear1 = nn.Linear(784, 128)      #First Layer: 784 -> 128
        self.relu1 = nn.ReLU()                  
        self.linear2 = nn.Linear(128, 64)       #Second Layer: 128 -> 64
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(64, 10)        #Output Layer: 64 -> 10 (One for each digit 0-9)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

if __name__ == "__main__":
    transform = transforms.ToTensor()
    model = SimpleMLP()
    BATCH_SIZE = 64

    train_dataset = create_train_dataset(transform)
    test_dataset = create_test_dataset(transform)
    train_dataloader = create_dataloaders(train_dataset, BATCH_SIZE, shuffle=True)
    test_dataloader = create_dataloaders(test_dataset, BATCH_SIZE, shuffle=False)
    
    get_single_data_batch(train_dataloader)
