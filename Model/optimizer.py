from abc import ABC, abstractmethod

import numpy as np

class Optimizer(ABC):
    """
    Abstract base class for optimizers.

    Methods:
        optim(self, grads, learning_rate): Update model parameters using gradients.
    """

    @abstractmethod
    def optim(self, grads, learning_rate):
        """
        Update model parameters using gradients.

        Args:
            grads (list): List of gradients for model parameters.
            learning_rate (float): Learning rate for the optimization.
        """
        pass


class SGDOptimizer(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Methods:
        optim(self, grads, learning_rate): Update model parameters using SGD.

    Attributes:
        None
    """

    def optim(self, grads, learning_rate):
        """
        Update model parameters using Stochastic Gradient Descent (SGD).

        Args:
            grads (list): List of gradients for model parameters.
            learning_rate (float): Learning rate for the optimization.
        """
        for layer_grads in grads:
            for i in range(len(layer_grads) - 1):
                params = layer_grads[i][0]
                grads = layer_grads[i][1]
                params -= learning_rate * grads



