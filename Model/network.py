import pickle
from abc import ABC

import numpy as np
from tqdm import tqdm

from Model.layers import ReLU, FC, Convolution, MaxPooling, Flatten
from utils import calculate_accuracy, calculate_macro_f1_score, calculate_micro_f1_score


class Model(ABC):
    """
    Abstract base class for neural network models.

    Attributes:
        layers (list): List of layers in the neural network.

    Methods:
        forward(x): Perform the forward pass of the neural network.
        backward(loss, x, y): Perform the backward pass of the neural network.
        add_layer(layer): Add a layer to the neural network.
        train(train_loader, val_loader, num_epochs, loss, optimizer, save_name, scheduler=None):
            Train the neural network using the specified training and validation data.
        test(test_loader, loss): Evaluate the neural network on a test dataset.
        save_parameters(filename): Save the model's parameters to a file.
        load_parameters(filename): Load the model's parameters from a file.
    """
    def __init__(self):
        """
        Initialize the neural network model.
        """

        self.layers = []

    def forward(self, x):
        """
        Perform the forward pass of the neural network.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Output of the neural network.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss, x, y):
        """
        Perform the backward pass of the neural network.

        Args:
            loss (object): Loss function used for training.
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): True labels.

        Returns:
            list: List of gradients for each layer.
        """
        grads = []
        grad_out = loss.backward_inputs(x, y)
        if loss.has_params:
            grads += loss.backward_params()

        for layer in reversed(self.layers):
            grad_inputs = layer.backward_inputs(grad_out)
            if layer.has_params:
                grads += [layer.backward_params(grad_out)]
            grad_out = grad_inputs
        return grads

    def add_layer(self, layer):
        """
        Add a layer to the neural network.

        Args:
            layer (object): Layer to be added.
        """
        self.layers.append(layer)

    def train(self, train_loader, val_loader, num_epochs, loss, optimizer, save_name, scheduler=None):
        """
        Train the neural network using the specified training and validation data.

        Args:
            train_loader (object): Data loader for training data.
            val_loader (object): Data loader for validation data.
            num_epochs (int): Number of training epochs.
            loss (object): Loss function used for training.
            optimizer (object): Optimizer for updating model parameters.
            save_name (str): Name for saving the best model weights.
            scheduler (object, optional): Learning rate scheduler. Defaults to None.

        Returns:
            tuple: A tuple containing lists of epoch accuracy, epoch loss, validation accuracies, and validation losses.
        """
        epoch_accuracy = []
        epoch_loss = []

        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []

        current_best_val_accuracy = 0.0

        for epoch in range(1, num_epochs + 1):

            cnt_correct = 0
            i = 0

            learning_rate = scheduler.get_lr()
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', ncols=100, colour="green")

            for batch_inputs, batch_targets in progress_bar:
                logits = self.forward(batch_inputs)
                loss_value = loss.forward(logits, batch_targets)
                yp = np.argmax(logits, 1)
                yt = np.argmax(batch_targets, 1)
                cnt_correct += (yp == yt).sum()
                grads = self.backward(loss, logits, batch_targets)
                optimizer.optim(grads=grads, learning_rate=learning_rate)

                accuracy = cnt_correct / ((i + 1) * len(batch_inputs))
                avg_loss = np.mean(loss_value)

                train_accuracies.append(accuracy)
                train_losses.append(avg_loss)

                if i % 10 == 0:
                    progress_bar.set_description(
                        f'Epoch {epoch}/{num_epochs}, Train Accuracy: {accuracy * 100:.3f}%, Train Loss: {avg_loss:.5f}')

                i += 1

            epoch_accuracy.append(np.mean(train_accuracies) * 100)
            epoch_loss.append(np.mean(train_losses))
            train_accuracies = []
            train_losses = []

            scheduler.step()
            val_accuracy, macro_f1, micro_f1, val_loss, _ = self.test(val_loader, loss)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print(
                f'Epoch {epoch}/{num_epochs}, Val Accuracy: {val_accuracy:.3f}%, Val Macro F1: {macro_f1:.3f}%, Val Micro F1: {micro_f1:.3f}%')

            if val_accuracy > current_best_val_accuracy:
                self.save_parameters(
                    f"ModelWeights/{save_name}_best_val_weights.pkl")

        return epoch_accuracy, epoch_loss, val_accuracies, val_losses

    def test(self, test_loader, loss):
        """
        Evaluate the neural network on a test dataset.

        Args:
            test_loader (object): Data loader for test data.
            loss (object): Loss function used for evaluation.

        Returns:
            tuple: A tuple containing accuracy, micro F1 score, macro F1 score, mean validation loss, and a list of wrong predictions.
        """
        pred_list = []
        target_list = []
        wrong_predictions = []
        val_losses = []

        for batch_inputs, batch_targets in test_loader:
            logits = self.forward(batch_inputs)
            preds = np.argmax(logits, 1)
            targets = np.argmax(batch_targets, 1)
            pred_list.extend(preds)
            target_list.extend(targets)

            # Check for incorrect predictions
            incorrect_mask = preds != targets
            incorrect_indices = np.where(incorrect_mask)[0]

            for idx in incorrect_indices:
                wrong_predictions.append((batch_inputs[idx], preds[idx], targets[idx]))

            loss_value = loss.forward(logits, batch_targets)
            val_losses.append(loss_value)

        preds = np.array(pred_list).astype(np.int64)
        targets = np.array(target_list).astype(np.int64)

        accuracy = calculate_accuracy(y_pred=preds, y_true=targets) * 100
        macro_f1_score = calculate_macro_f1_score(y_pred=preds, y_true=targets) * 100
        micro_f1_score = calculate_micro_f1_score(y_pred=preds, y_true=targets) * 100

        return accuracy, micro_f1_score, macro_f1_score, np.mean(val_losses), wrong_predictions

    def save_parameters(self, filename):
        """
        Save the model's parameters to a file.

        Args:
            filename (str): Name of the file to save the parameters to.
        """
        parameters = {}
        for layer in self.layers:
            if layer.has_params:
                parameters[layer.name] = (layer.weights, layer.bias)
        with open(filename, 'wb') as file:
            pickle.dump(parameters, file)

    def load_parameters(self, filename):
        """
        Load the model's parameters from a file.

        Args:
            filename (str): Name of the file to load the parameters from.
        """
        with open(filename, 'rb') as file:
            parameters = pickle.load(file)
        for layer in self.layers:
            if layer.has_params:
                layer.weights, layer.bias = parameters[layer.name]


class MNIST_classifier(Model):

    def __init__(self, input):
        """
        Neural network model for MNIST classification.

        Attributes:
            name (str): Name of the model.

        Methods:
            __init__(self, input): Initialize the model with specified input size.
        """
        super().__init__()
        self.name = self.__class__.__name__

        # Define the layers of the model
        self.add_layer(FC(input, 500, "fc1"))
        self.add_layer(ReLU(self.layers[-1], "relu1"))
        self.add_layer(FC(self.layers[-1], 500, "fc2"))
        self.add_layer(ReLU(self.layers[-1], "relu2"))
        self.add_layer(FC(self.layers[-1], 500, "fc3"))
        self.add_layer(ReLU(self.layers[-1], "relu3"))
        self.add_layer(FC(self.layers[-1], 10, "logits"))


class MNIST_classifier_convolution(Model):

    def __init__(self, input, dropout_rate=0.0):
        """
        Convolutional neural network model for MNIST classification.

        Attributes:
            name (str): Name of the model.
            dropout_rate (float): Dropout rate for the fully connected layer.

        Methods:
            __init__(self, input, dropout_rate=0.0): Initialize the model with specified input size and dropout rate.
        """
        super().__init__()
        self.name = self.__class__.__name__
        self.droupout_rate = dropout_rate

        # Define the layers of the model
        self.add_layer(Convolution(input, 16, 5, name="conv1"))
        self.add_layer(MaxPooling(self.layers[-1], "pool1"))
        self.add_layer(ReLU(self.layers[-1], "relu1"))

        self.add_layer(Convolution(self.layers[-1], 32, 5, name="conv2"))
        self.add_layer(MaxPooling(self.layers[-1], "pool2"))
        self.add_layer(ReLU(self.layers[-1], "relu2"))

        self.add_layer(Flatten(self.layers[-1], "flatten3"))
        self.add_layer(FC(self.layers[-1], 512, dropout_rate=self.droupout_rate, name="fc3"))

        self.add_layer(ReLU(self.layers[-1], "relu3"))
        self.add_layer(FC(self.layers[-1], 10, "logits"))
