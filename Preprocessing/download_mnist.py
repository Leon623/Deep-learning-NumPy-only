import os

from sklearn.datasets import fetch_openml

if __name__ == "__main__":
    mnist_folder = '../mnist'

    if not os.path.exists(mnist_folder):
        os.makedirs(mnist_folder)

    mnist = fetch_openml('mnist_784', version=1, data_home=mnist_folder)

    data_filename = os.path.join(mnist_folder, 'mnist_data.csv')
    labels_filename = os.path.join(mnist_folder, 'mnist_labels.csv')

    mnist.data.to_csv(data_filename, index=False)
    mnist.target.to_csv(labels_filename, index=False)

    print(f"MNIST dataset downloaded and saved to '{mnist_folder}' folder.")
