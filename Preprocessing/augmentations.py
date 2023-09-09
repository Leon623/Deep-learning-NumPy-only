import numpy as np
from scipy.ndimage import rotate
import random

class Transforms:
    @staticmethod
    def horizontal_flip(image, p=0.5):
        if np.random.rand() < p:
            image = np.fliplr(image)
        return image

    @staticmethod
    def vertical_flip(image, p=0.5):
        if np.random.rand() < p:
            image = np.flipud(image)
        return image

    @staticmethod
    def random_rotation(image, max_angle=15, p=0.5):
        if np.random.rand() < p:
            angle = np.random.uniform(-max_angle, max_angle)
            image = rotate(image, angle, reshape=False)
        return image


class Augmentator:
    def __init__(self, transforms=None):
        self.transforms = transforms if transforms is not None else []

    def __call__(self, image):
        for transform, p in self.transforms:
            image = transform(image, p)
        return image