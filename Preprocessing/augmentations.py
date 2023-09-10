import numpy as np
from scipy.ndimage import rotate


class Transforms:
    """A collection of image transformation functions."""

    @staticmethod
    def horizontal_flip(image, p=0.5):
        """
        Apply horizontal flip to an image with a given probability.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.
            p (float, optional): Probability of applying the transformation. Defaults to 0.5.

        Returns:
            numpy.ndarray: Transformed image.
        """
        if np.random.rand() < p:
            image = np.fliplr(image)
        return image

    @staticmethod
    def vertical_flip(image, p=0.5):
        """
        Apply vertical flip to an image with a given probability.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.
            p (float, optional): Probability of applying the transformation. Defaults to 0.5.

        Returns:
            numpy.ndarray: Transformed image.
        """
        if np.random.rand() < p:
            image = np.flipud(image)
        return image

    @staticmethod
    def random_rotation(image, max_angle=15, p=0.5):
        """
        Apply random rotation to an image with a given probability.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.
            max_angle (float, optional): Maximum angle for rotation in degrees. Defaults to 15.
            p (float, optional): Probability of applying the transformation. Defaults to 0.5.

        Returns:
            numpy.ndarray: Transformed image.
        """
        if np.random.rand() < p:
            angle = np.random.uniform(-max_angle, max_angle)
            image = rotate(image, angle, reshape=False)
        return image


class Augmentator:
    """Class for applying a sequence of image transformations."""

    def __init__(self, transforms=None):
        """
        Initialize an Augmentator.

        Args:
            transforms (list, optional): List of transformation functions with probabilities.
        """
        self.transforms = transforms if transforms is not None else []

    def __call__(self, image):
        """
        Apply a sequence of image transformations to the input image.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.

        Returns:
            numpy.ndarray: Transformed image.
        """
        for transform, p in self.transforms:
            image = transform(image, p)
        return image
