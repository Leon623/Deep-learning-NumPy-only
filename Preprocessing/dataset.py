from abc import ABC, abstractmethod


class CustomDataset(ABC):
    """Abstract base class for custom datasets."""

    def __init__(self, data, targets, transform=None):
        """
        Initialize a CustomDataset.

        Args:
            data (list or numpy array): The dataset samples.
            targets (list or numpy array): The corresponding targets.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    @abstractmethod
    def __len__(self):
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return the sample and its target for the given index."""
        pass


class MNISTDataset(CustomDataset):
    """Dataset class for MNIST dataset."""

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the sample and its target for the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the sample and its target.
        """
        sample = self.data[idx], self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
