from abc import ABC, abstractmethod

import numpy as np
import scipy.stats as stats
import numpy as np
from time import time
import sys

sys.path.append('..')
from im2col_cython import col2im_cython, im2col_cython

zero_init = np.zeros


def variance_scaling_initializer(shape, fan_in, factor=2.0):
    sigma = np.sqrt(factor / fan_in)
    return stats.truncnorm(-2, 2, loc=0, scale=sigma).rvs(shape)

class Layer(ABC):
    """
    Abstract base class for neural network layers.

    Attributes:
        has_params (bool): Indicates whether the layer has learnable parameters.
    """

    @abstractmethod
    def forward(self, inputs):
        """
        Forward pass of the layer.

        Args:
            inputs (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the layer.
        """
        pass

    @abstractmethod
    def backward_inputs(self, grads):
        """
        Backward pass for computing gradients with respect to inputs.

        Args:
            grads (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        pass

    def backward_params(self, grads):
        """
        Backward pass for computing gradients with respect to layer parameters.

        Args:
            grads (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            None: This method can be overridden in subclasses if the layer has parameters.
        """
        pass

class Flatten(Layer):
    """
    Flatten layer that reshapes input to 1D.

    Attributes:
        has_params (bool): Indicates whether the layer has learnable parameters.
    """

    def __init__(self, input_layer, name):
        """
        Initialize a Flatten layer.

        Args:
            input_layer (Layer): Input layer.
            name (str): Layer name.
        """
        self.input_shape = input_layer.shape
        self.N = self.input_shape[0]
        self.num_outputs = 1
        for i in range(1, len(self.input_shape)):
            self.num_outputs *= self.input_shape[i]
        self.shape = (self.N, self.num_outputs)
        self.has_params = False
        self.name = name

    def forward(self, inputs):
        """
        Forward pass of the Flatten layer.

        Args:
            inputs (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Flattened output.
        """
        self.input_shape = inputs.shape
        inputs_flat = inputs.reshape(self.input_shape[0], -1)
        self.shape = inputs_flat.shape
        return inputs_flat

    def backward_inputs(self, grads):
        """
        Backward pass for computing gradients with respect to inputs.

        Args:
            grads (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        return grads.reshape(self.input_shape)



class FC(Layer):
    """
    Fully connected (dense) layer.

    Attributes:
        input_shape (tuple): Shape of the input data.
        N (int): Number of samples in the input.
        shape (tuple): Shape of the layer's output.
        num_outputs (int): Number of output units.
        dropout_rate (float): Dropout rate (between 0 and 1).
        dropout_mask (numpy.ndarray): Stores the dropout mask.
        num_inputs (int): Number of input features.
        weights (numpy.ndarray): Weights of the layer.
        bias (numpy.ndarray): Bias of the layer.
        name (str): Layer name.
        has_params (bool): Indicates whether the layer has learnable parameters.
    """

    def __init__(self, input_layer, num_outputs, name,
                 weights_initializer_fn=variance_scaling_initializer,
                 bias_initializer_fn=zero_init, dropout_rate=0.0):
        """
        Initialize a Fully Connected (FC) layer.

        Args:
            input_layer (Layer): Input layer.
            num_outputs (int): Number of output units.
            name (str): Layer name.
            weights_initializer_fn (function): Function for initializing weights.
            bias_initializer_fn (function): Function for initializing bias.
            dropout_rate (float, optional): Dropout rate (between 0 and 1). Defaults to 0.0.
        """
        self.input_shape = input_layer.shape
        self.N = self.input_shape[0]
        self.shape = (self.N, num_outputs)
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        self.num_inputs = 1
        for i in range(1, len(self.input_shape)):
            self.num_inputs *= self.input_shape[i]
        self.weights = weights_initializer_fn([num_outputs, self.num_inputs], fan_in=self.num_inputs)
        self.bias = bias_initializer_fn([num_outputs])
        self.name = name
        self.has_params = True

    def forward(self, inputs, is_training=True):
        """
        Forward pass of the Fully Connected (FC) layer.

        Args:
            inputs (numpy.ndarray): Input data.
            is_training (bool, optional): Indicates if the network is in training mode. Defaults to True.

        Returns:
            numpy.ndarray: Output of the layer.
        """
        self.input = inputs

        if is_training and self.dropout_rate > 0:
            self.dropout_mask = np.random.rand(*inputs.shape) > self.dropout_rate
            inputs *= self.dropout_mask / (1 - self.dropout_rate)

        return inputs @ self.weights.T + self.bias

    def backward_inputs(self, grads):
        """
        Backward pass for computing gradients with respect to inputs.

        Args:
            grads (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        if self.dropout_mask is not None:
            dropout_mask = self.dropout_mask[:, :grads.shape[1]]
            grads *= dropout_mask / (1 - self.dropout_rate)
        return grads @ self.weights

    def backward_params(self, grads):
        """
        Backward pass for computing gradients with respect to layer parameters.

        Args:
            grads (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            list: List containing weight and bias gradients, and the layer name.
        """
        grad_weights = grads.T @ self.input
        grad_bias = np.sum(grads, axis=0)
        return [[self.weights, grad_weights], [self.bias, grad_bias], self.name]


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) activation layer.

    Attributes:
        shape (tuple): Shape of the input data.
        name (str): Layer name.
        has_params (bool): Indicates whether the layer has learnable parameters.
    """

    def __init__(self, input_layer, name):
        """
        Initialize a ReLU activation layer.

        Args:
            input_layer (Layer): Input layer.
            name (str): Layer name.
        """
        self.shape = input_layer.shape
        self.name = name
        self.has_params = False

    def forward(self, inputs):
        """
        Forward pass of the ReLU activation layer.

        Args:
            inputs (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the layer.
        """
        self.input = inputs
        inputs[inputs < 0] = 0
        return inputs

    def backward_inputs(self, grads):
        """
        Backward pass for computing gradients with respect to inputs.

        Args:
            grads (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        inputs = self.input.copy()
        return grads * (inputs > 0)


class Convolution(Layer):
    """
    Convolutional layer.

    Attributes:
        input_shape (tuple): Shape of the input data.
        C (int): Number of input channels.
        N (int): Number of samples in the input.
        num_filters (int): Number of convolutional filters.
        kernel_size (int): Size of the convolutional kernel.
        padding (str): Padding mode, 'SAME' or 'VALID'.
        shape (tuple): Shape of the layer's output.
        pad (int): Padding size.
        weights (numpy.ndarray): Weights of the layer.
        bias (numpy.ndarray): Bias of the layer.
        stride (int): Stride for convolution operation.
        name (str): Layer name.
        has_params (bool): Indicates whether the layer has learnable parameters.
    """

    def __init__(self, input_layer, num_filters, kernel_size, name, padding='SAME',
                 weights_initializer_fn=variance_scaling_initializer,
                 bias_initializer_fn=zero_init):
        """
        Initialize a Convolutional layer.

        Args:
            input_layer (Layer): Input layer.
            num_filters (int): Number of convolutional filters.
            kernel_size (int): Size of the convolutional kernel.
            name (str): Layer name.
            padding (str, optional): Padding mode, 'SAME' or 'VALID'. Defaults to 'SAME'.
            weights_initializer_fn (function): Function for initializing weights.
            bias_initializer_fn (function): Function for initializing bias.
        """
        self.input_shape = input_layer.shape
        N, C, H, W = input_layer.shape
        self.C = C
        self.N = N
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1

        self.padding = padding
        if padding == 'SAME':
            self.shape = (N, num_filters, H, W)
            self.pad = (kernel_size - 1) // 2
        else:
            self.shape = (N, num_filters, H - kernel_size + 1, W - kernel_size + 1)
            self.pad = 0

        fan_in = C * kernel_size**2
        self.weights = weights_initializer_fn([num_filters, kernel_size**2 * C], fan_in)
        self.bias = bias_initializer_fn([num_filters])
        self.stride = 1
        self.name = name
        self.has_params = True

    def forward(self, x):
        """
        Forward pass of the Convolutional layer.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the layer.
        """
        k = self.kernel_size
        self.x_cols = im2col_cython(x, k, k, self.pad, self.stride)
        res = self.weights.dot(self.x_cols) + self.bias.reshape(-1, 1)
        N, C, H, W = x.shape
        out = res.reshape(self.num_filters, self.shape[2], self.shape[3], N)
        return out.transpose(3, 0, 1, 2)

    def backward_inputs(self, grad_out):
        """
        Backward pass for computing gradients with respect to inputs.

        Args:
            grad_out (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        grad_out = grad_out.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        grad_x_cols = self.weights.T.dot(grad_out)
        N, C, H, W = self.input_shape
        k = self.kernel_size
        grad_x = col2im_cython(grad_x_cols, N, C, H, W, k, k, self.pad, self.stride)
        return grad_x

    def backward_params(self, grad_out):
        """
        Backward pass for computing gradients with respect to layer parameters.

        Args:
            grad_out (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            list: List containing weight and bias gradients, and the layer name.
        """
        grad_bias = np.sum(grad_out, axis=(0, 2, 3))
        grad_out = grad_out.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        grad_weights = grad_out.dot(self.x_cols.T).reshape(self.weights.shape)
        return [[self.weights, grad_weights], [self.bias, grad_bias], self.name]


class MaxPooling(Layer):
    """
    Max Pooling layer.

    Attributes:
        name (str): Layer name.
        input_shape (tuple): Shape of the input data.
        stride (int): Stride for pooling operation.
        shape (tuple): Shape of the layer's output.
        pool_size (int): Size of the pooling window.
        has_params (bool): Indicates whether the layer has learnable parameters.
    """

    def __init__(self, input_layer, name, pool_size=2, stride=2):
        """
        Initialize a Max Pooling layer.

        Args:
            input_layer (Layer): Input layer.
            name (str): Layer name.
            pool_size (int, optional): Size of the pooling window. Defaults to 2.
            stride (int, optional): Stride for pooling operation. Defaults to 2.
        """
        self.name = name
        self.input_shape = input_layer.shape
        N, C, H, W = self.input_shape
        self.stride = stride
        self.shape = (N, C, H // stride, W // stride)
        self.pool_size = pool_size
        assert pool_size == stride, 'Invalid pooling params'
        assert H % pool_size == 0
        assert W % pool_size == 0
        self.has_params = False

    def forward(self, x):
        """
        Forward pass of the Max Pooling layer.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the layer.
        """
        N, C, H, W = x.shape
        self.input_shape = x.shape
        self.x = x.reshape(N, C, H // self.pool_size, self.pool_size,
                           W // self.pool_size, self.pool_size)
        self.out = self.x.max(axis=3).max(axis=4)
        return self.out.copy()

    def backward_inputs(self, grad_out):
        """
        Backward pass for computing gradients with respect to inputs.

        Args:
            grad_out (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        grad_x = np.zeros_like(self.x)
        out_newaxis = self.out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (self.x == out_newaxis)
        dout_newaxis = grad_out[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, grad_x)
        grad_x[mask] = dout_broadcast[mask]
        grad_x /= np.sum(mask, axis=(3, 5), keepdims=True)
        grad_x = grad_x.reshape(self.input_shape)
        return grad_x
