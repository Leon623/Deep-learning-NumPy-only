from abc import ABC, abstractmethod


class CustomDataset(ABC):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class MNISTDataset(CustomDataset):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx], self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
