from abc import ABC, abstractmethod
from Preprocessing.augmentations import Augmentator
import numpy as np


class CustomDataLoader(ABC):
    def __init__(self, dataset, batch_size=1, shuffle=False, augmentator=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))
        self.augmentator = augmentator

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class MNISTDataLoader(CustomDataLoader):
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start: start + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            batch_inputs, batch_targets = zip(*batch_data)

            if self.augmentator:
                batch_inputs = [self.augmentator(np.array(input_image)) for input_image in batch_inputs]

            yield np.array(batch_inputs), np.array(batch_targets)


    def __len__(self):
        return len(self.dataset) // self.batch_size + int(len(self.dataset) % self.batch_size != 0)
