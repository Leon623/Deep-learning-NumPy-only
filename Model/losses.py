import numpy as np


class SoftmaxCrossEntropyWithLogits:
    """
    Softmax Cross Entropy loss function.

    This class computes the Softmax Cross Entropy loss and its gradients with respect to the logits.

    Attributes:
        has_params (bool): Indicates whether the loss function has learnable parameters.
    """

    def __init__(self):
        """
        Initialize the Softmax Cross Entropy loss function.
        """
        self.has_params = False

    def forward(self, x, y):
        """
        Compute the forward pass of the Softmax Cross Entropy loss.

        Args:
            x (numpy.ndarray): Logits from the model.
            y (numpy.ndarray): True labels (one-hot encoded).

        Returns:
            numpy.ndarray: Softmax Cross Entropy loss.
        """
        stable_x = np.exp(x - x.max())
        softmax = stable_x / np.sum(stable_x, axis=1, keepdims=True)
        return -np.sum(np.log(softmax) * y, axis=1) / x.shape[0]

    def backward_inputs(self, x, y):
        """
        Compute the gradients of the Softmax Cross Entropy loss with respect to the logits.

        Args:
            x (numpy.ndarray): Logits from the model.
            y (numpy.ndarray): True labels (one-hot encoded).

        Returns:
            numpy.ndarray: Gradients of the loss with respect to the logits.
        """
        stable_x = np.exp(x - x.max())
        softmax = stable_x / np.sum(stable_x, axis=1, keepdims=True)
        return (softmax - y) / x.shape[0]
