import numpy as np


class SoftmaxCrossEntropyWithLogits():
    def __init__(self):
        self.has_params = False

    def forward(self, x, y):
        stable_x = np.exp(x - x.max())
        softmax = stable_x / np.sum(stable_x, axis=1, keepdims=True)
        return -np.sum(np.log(softmax) * y, axis=1) / x.shape[0]

    def backward_inputs(self, x, y):
        stable_x = np.exp(x - x.max())
        softmax = stable_x / np.sum(stable_x, axis=1, keepdims=True)
        return (softmax - y) / x.shape[0]