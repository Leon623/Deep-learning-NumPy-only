import pickle
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

from Model.layers import ReLU, FC, Convolution, MaxPooling, Flatten
from utils import calculate_accuracy, calculate_macro_f1_score, calculate_micro_f1_score


class Model(ABC):
    def __init__(self):
        self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss, x, y):
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
        self.layers.append(layer)

    def train(self, train_loader, val_loader, num_epochs, loss, optimizer, scheduler=None):

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

            epoch_accuracy.append(np.mean(train_accuracies)*100)
            epoch_loss.append(np.mean(train_losses))
            train_accuracies = []
            train_losses = []

            scheduler.step()
            val_accuracy, macro_f1, micro_f1, val_loss = self.test(val_loader, loss)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print(
                f'Epoch {epoch}/{num_epochs}, Val Accuracy: {val_accuracy:.3f}%, Val Macro F1: {macro_f1:.3f}%, Val Micro F1: {micro_f1:.3f}%')
            if val_accuracy > current_best_val_accuracy:
                self.save_parameters(f"ModelWeights/{self.name}_{epoch}_weights.pkl")

        return epoch_accuracy, epoch_loss, val_accuracies, val_losses

    def test(self, test_loader, loss):

        pred_list = []
        target_list = []

        val_losses = []

        for batch_inputs, batch_targets in test_loader:
            logits = self.forward(batch_inputs)
            preds = np.argmax(logits, 1)
            targets = np.argmax(batch_targets, 1)
            pred_list.extend(preds)
            target_list.extend(targets)
            loss_value = loss.forward(logits, batch_targets)
            val_losses.append(loss_value)

        preds = np.array(pred_list).astype(np.int64)
        targets = np.array(target_list).astype(np.int64)

        accuracy = calculate_accuracy(y_pred=preds, y_true=targets) * 100
        macro_f1_score = calculate_macro_f1_score(y_pred=preds, y_true=targets) * 100
        micro_f1_score = calculate_micro_f1_score(y_pred=preds, y_true=targets) * 100

        return accuracy, micro_f1_score, macro_f1_score, np.mean(val_losses)

    def save_parameters(self, filename):
        parameters = {}
        for layer in self.layers:
            if layer.has_params:
                parameters[layer.name] = (layer.weights, layer.bias)
        with open(filename, 'wb') as file:
            pickle.dump(parameters, file)

    def load_parameters(self, filename):
        with open(filename, 'rb') as file:
            parameters = pickle.load(file)
        for layer in self.layers:
            if layer.has_params:
                layer.weights, layer.bias = parameters[layer.name]


class MNIST_classifier(Model):

    def __init__(self, input):
        super().__init__()

        # Define the layers of the model
        self.add_layer(FC(input, 500, "fc1"))
        self.add_layer(ReLU(self.layers[-1], "relu1"))
        self.add_layer(FC(self.layers[-1], 500, "fc2"))
        self.add_layer(ReLU(self.layers[-1], "relu2"))
        self.add_layer(FC(self.layers[-1], 500, "fc3"))
        self.add_layer(ReLU(self.layers[-1], "relu3"))
        self.add_layer(FC(self.layers[-1], 10, "logits"))


class MNIST_classifier_convolution(Model):

    def __init__(self, input):
        super().__init__()
        self.name = self.__class__.__name__

        self.add_layer(Convolution(input, 32, 5, padding=2, name="conv1"))
        self.add_layer(MaxPooling(self.layers[-1], "pool1"))
        self.add_layer(ReLU(self.layers[-1], "relu1"))

        self.add_layer(Convolution(self.layers[-1], 32, 5, name="conv2", padding=2))
        self.add_layer(MaxPooling(self.layers[-1], "pool2"))
        self.add_layer(ReLU(self.layers[-1], "relu2"))

        self.add_layer(Flatten(self.layers[-1], "flatten3"))
        self.add_layer(FC(self.layers[-1], 512, "fc3"))
        self.add_layer(ReLU(self.layers[-1], "relu3"))
        self.add_layer(FC(self.layers[-1], 10, "logits"))
