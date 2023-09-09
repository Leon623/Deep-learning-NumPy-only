from abc import ABC, abstractmethod

import numpy as np
class Optimizer(ABC):
    @abstractmethod
    def optim(self, grads, learning_rate):
        pass


class SGDOptimizer(Optimizer):
    def optim(self, grads, learning_rate):
        for layer_grads in grads:
            for i in range(len(layer_grads) - 1):
                params = layer_grads[i][0]
                grads = layer_grads[i][1]
                params -= learning_rate * grads


