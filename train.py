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

def main(args):

    # Parsing the arguments
    data_dir = args.data_path
    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    random_state = args.random_state
    augentations = args.augmentations

    # Files which ontains data and labels
    data_path = f'{data_dir}/mnist_data.csv'
    labels_path = f'{data_dir}/mnist_labels.csv'

    # Reading and reshaping th data
    data = (pd.read_csv(data_path).values.astype(np.float32) / 255.0)
    data = data.reshape(-1, 1, 28, 28)
    # Reading and one-hot encoding the target values
    targets = pd.read_csv(labels_path).values.astype(np.int64)
    targets = one_hot_encode_list(targets)

    # Splitting data into train validation and test set ( 70% train 10% validation 20% test)
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125)

    train_dataset = MNISTDataset(X_train, y_train)
    val_dataset = MNISTDataset(X_val, y_val)
    test_dataset = MNISTDataset(X_test, y_test)

    # Adding augmentations if needed
    augmentator = None
    if augentations:
        transforms = [(Transforms.random_rotation, 0.2)]
        augmentator = Augmentator(transforms=transforms)

    # Creating loaders for the datasets
    train_loader = MNISTDataLoader(train_dataset, batch_size=batch_size, shuffle=True, augmentator=augmentator)
    val_loader = MNISTDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = MNISTDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"Train dataset size: {len(train_dataset)}\nVal dataset size: {len(val_dataset)}\nTest dataset size: {len(test_dataset)}")

    # Initializing scheduler
    scheduler = LinearScheduler(initial_lr=lr)
    # Initializing optimizer
    optimizer = SGDOptimizer()
    # Initializing loss
    loss = SoftmaxCrossEntropyWithLogits()

    # Shape of inputs for network to know which kind of data to expect
    inputs = np.expand_dims(data[0], axis=0)

    # Initializing the model
    model = MNIST_classifier_convolution(inputs)

    # Training the model
    train_accuracies, train_losses, val_accuracies, val_losses = model.train(loss=loss, num_epochs=num_epochs,
                                                                           train_loader=train_loader,
                                                                           scheduler=scheduler,
                                                                           val_loader=val_loader, optimizer=optimizer)

    # Plotting the performances on the train and validation dataset
    plot_performance(train_accuracies, train_losses, val_accuracies, val_losses)

    # Loading the weights from the best validation epoch score
    model.load_parameters(filename="ModelWeights/best_val_accuracy_weights.pkl")

    # Testing the model
    accuracy, micro_f1, macro_f1, val_losses = model.test(test_loader=test_loader, loss=loss)
    print("-----------------------")
    print(f"Test accuracy: {accuracy:.3f}%\nTest micro-f1: {micro_f1:.3f}%\nTest macro-f1: {macro_f1:.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-batch_size", type=int, default=100,
                        help="Batch size of train, test and val, must be the divider of 49000, 7000, 14000!")
    parser.add_argument("-num_epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("-random_state", type=int, default=42, help="Random state of the test_val split")
    parser.add_argument("-data_path", type=str, default="mnist",
                        help="Path to the folder containing mnist_data.csv and mnist_labels.csv")
    parser.add_argument("-lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("-augmentations", action="store_true", help="Do you want to use augmentations for training")

    args = parser.parse_args()
    main(args)
