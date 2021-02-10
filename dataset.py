import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

mean = {
    'MNIST': np.array([0.1307]),
    'FashionMNIST': np.array([0.2860]),
    'CIFAR10': np.array([0.4914, 0.4822, 0.4465]),
}
std = {
    'MNIST': 0.3081,
    'FashionMNIST': 0.3520,
    'CIFAR10': 0.2009, #np.array([0.2023, 0.1994, 0.2010])
}
train_transforms = {
    'MNIST': [transforms.RandomCrop(28, padding=1)],
    'FashionMNIST': [transforms.RandomCrop(28, padding=1)],
    'CIFAR10': [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()],
}
input_dim = {
    'MNIST': np.array([1, 28, 28]),
    'FashionMNIST': np.array([1, 28, 28]),
    'CIFAR10': np.array([3, 32, 32]),
}
default_eps = {
    'MNIST': 0.3,
    'FashionMNIST': 0.1,
    'CIFAR10': 0.03137,
}

def get_statistics(dataset):
    return mean[dataset], std[dataset]

class TruncateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.classes = [int(i) for i in classes]
        # only load targets which will be faster
        self.indexes = [i for i in range(len(dataset)) if dataset.targets[i] in self.classes]
    def __getattr__(self, item):
        return getattr(self.dataset, item)
    def __getitem__(self, index):
        item = self.dataset[self.indexes[index]]
        return item[0], self.classes.index(item[1])
    def __len__(self):
        return len(self.indexes)

def get_dataset(dataset, datadir, augmentation=True, classes=None):
    default_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset], [std[dataset]] * len(mean[dataset]))
    ]
    train_transform = transforms.Compose((train_transforms[dataset] if augmentation else []) + default_transform)
    test_transform = transforms.Compose(default_transform)
    Dataset = globals()[dataset]
    train_dataset = Dataset(root=datadir, train=True, download=True, transform=train_transform)
    test_dataset = Dataset(root=datadir, train=False, download=True, transform=test_transform)
    if classes is not None:
        train_dataset = TruncateDataset(train_dataset, classes)
        test_dataset = TruncateDataset(test_dataset, classes)
    return train_dataset, test_dataset

def load_data(dataset, datadir, batch_size, parallel, augmentation=True, workers=2, classes=None):
    train_dataset, test_dataset = get_dataset(dataset, datadir, augmentation=augmentation, classes=classes)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if parallel else None
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                             num_workers=workers, sampler=train_sampler, pin_memory=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if parallel else None
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, sampler=test_sampler, pin_memory=True)
    return trainloader, testloader

