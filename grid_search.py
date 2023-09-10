import argparse
import itertools

import numpy as np
import pandas as pd

from Model.losses import SoftmaxCrossEntropyWithLogits
from Model.network import MNIST_classifier_convolution
from Model.optimizer import SGDOptimizer
from Model.scheduler import LinearScheduler
from Preprocessing.dataloader import MNISTDataLoader
from Preprocessing.dataset import MNISTDataset
from utils import one_hot_encode_list
from utils import plot_performance
from utils import train_test_split


def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_floats(arg):
    return list(map(float, arg.split(',')))


def main(args):
    # Parsing arguments
    data_dir = args.data_path
    random_state = args.random_state

    # Initializing grid for grid search
    grid = {

        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'num_epoch': args.num_epoch
    }

    # Files which contains data and labels
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

    # Initializing optimizer
    optimizer = SGDOptimizer()
    # Initializing loss
    loss = SoftmaxCrossEntropyWithLogits()
    # Shape of inputs for network to know which kind of data to expect
    inputs = np.expand_dims(data[0], axis=0)

    # Grid search loop
    try:
        for values in itertools.product(*grid.values()):
            point = dict(zip(grid.keys(), values))

            batch_size = point['batch_size']
            lr = point['learning_rate']
            num_epochs = point['num_epoch']
            droupout_rate = point['dropout_rate']

            # merge the general settings
            settings = {**point}
            print(settings)

            # Ensure that all batches are the same size, this is important for network to work!
            assert len(train_dataset) % batch_size == len(val_dataset) % batch_size == len(test_dataset) % batch_size == 0

            # Creating loaders for the datasets
            train_loader = MNISTDataLoader(train_dataset, batch_size=batch_size, shuffle=True, augmentator=None)
            val_loader = MNISTDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = MNISTDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            print(
                f"Train dataset size: {len(train_dataset)}\nVal dataset size: {len(val_dataset)}\nTest dataset size: {len(test_dataset)}",
                flush=True)

            # Initializing the model
            model = MNIST_classifier_convolution(inputs)
            # Initializing scheduler
            scheduler = LinearScheduler(initial_lr=lr)

            # How will the model be saved, for grid search
            save_name = f"{model.name}_{num_epochs}_{batch_size}_{lr}_{droupout_rate}_{random_state}"

            # Training the model
            train_accuracies, train_losses, val_accuracies, val_losses = model.train(loss=loss, num_epochs=num_epochs,
                                                                                     train_loader=train_loader,
                                                                                     scheduler=scheduler,
                                                                                     val_loader=val_loader,
                                                                                     optimizer=optimizer,
                                                                                     save_name=save_name)
            # Plotting the performances on the train and validation dataset
            plot_performance(train_accuracies, train_losses, val_accuracies, val_losses, save_name)

            # Save results to file
            with open("grid_search.txt", "a") as f:
                f.write(f"{save_name}:{max(val_accuracies)}\n")

            # Loading the weights from the best validation epoch score
            model.load_parameters(
                filename=f"ModelWeights/{save_name}_best_val_weights.pkl")

            # Testing the model on the test set
            accuracy, micro_f1, macro_f1, val_losses, _ = model.test(test_loader=test_loader, loss=loss)
            print("-----------------------")
            print(f"Test accuracy: {accuracy:.3f}%\nTest micro-f1: {micro_f1:.3f}%\nTest macro-f1: {macro_f1:.3f}%")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch_size', type=list_of_ints, help="List of batch sizes")
    parser.add_argument('-learning_rate', type=list_of_floats, help="List of learning rates")
    parser.add_argument('-dropout_rate', type=list_of_floats, help="List of dropout rates")
    parser.add_argument('-num_epoch', type=list_of_ints, help="List of num epochs")
    parser.add_argument("-data_path", type=str, default="mnist",
                        help="Path to the folder containing mnist_data.csv and mnist_labels.csv")
    parser.add_argument("-random_state", type=int, default=42, help="Random state of the test_val split")

    args = parser.parse_args()

    main(args)
