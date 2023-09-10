import argparse

import numpy as np
import pandas as pd

from Model.losses import SoftmaxCrossEntropyWithLogits
from Model.network import MNIST_classifier_convolution
from Model.optimizer import SGDOptimizer
from Model.scheduler import LinearScheduler
from Preprocessing.augmentations import Augmentator, Transforms
from Preprocessing.dataloader import MNISTDataLoader
from Preprocessing.dataset import MNISTDataset
from utils import one_hot_encode_list
from utils import plot_performance
from utils import train_test_split
from utils import plot_wrong_predictions
from utils import plot_class_errors

def main(args):

    batch_size = args.batch_size
    data_dir = args.data_path
    random_state = args.random_state
    weights_path = args.weights_path

    # Files which ontains data and labels
    data_path = f'{data_dir}/mnist_data.csv'
    labels_path = f'{data_dir}/mnist_labels.csv'

    # Reading and reshaping th data
    data = (pd.read_csv(data_path).values.astype(np.float32) / 255.0)
    data = data.reshape(-1, 1, 28, 28)
    # Reading and one-hot encoding the target values
    targets = pd.read_csv(labels_path).values.astype(np.int64)
    test_labels = targets.ravel()
    targets = one_hot_encode_list(targets)

    # Splitting the data to get the test dataset, use the same random split state from training!
    _, X_test, _, y_test = train_test_split(data, targets, test_size=0.2, random_state=random_state)
    test_dataset = MNISTDataset(X_test, y_test)

    # Creating the test loader
    test_loader = MNISTDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Test dataset size: {len(test_dataset)}")


    # Shape of inputs for network to know which kind of data to expect
    inputs = np.expand_dims(data[0], axis=0)

    # Initializing the model
    model = MNIST_classifier_convolution(inputs)

    # Loading the weights from the weights path
    model.load_parameters(filename=weights_path)

    # Initializing loss
    loss = SoftmaxCrossEntropyWithLogits()

    # Testing the model
    accuracy, micro_f1, macro_f1, _, wrong_predictions = model.test(test_loader=test_loader, loss=loss)
    print("-----------------------")
    print(f"Test accuracy: {accuracy:.3f}%\nTest micro-f1: {micro_f1:.3f}%\nTest macro-f1: {macro_f1:.3f}%")
    plot_wrong_predictions(wrong_predictions=wrong_predictions[:10])
    plot_class_errors(wrong_predictions=wrong_predictions, test_labels=test_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-batch_size", type=int, default=100,
                        help="Batch size of train, test and val, must be the divider of 49000, 7000, 14000!")
    parser.add_argument("-random_state", type=int, default=42, help="Random state of the test_val split")
    parser.add_argument("-data_path", type=str, default="mnist",
                        help="Path to the folder containing mnist_data.csv and mnist_labels.csv")
    parser.add_argument("-weights_path", type=str, default="ModelWeights/MNIST_classifier_convolution_8_best_val_weights.pkl",
                        help="Path to the trained model weights")

    args = parser.parse_args()
    main(args)
