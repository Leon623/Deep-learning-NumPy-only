import matplotlib.pyplot as plt
import numpy as np


def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def calculate_macro_f1_score(y_true, y_pred):
    f1_scores = []
    for class_label in np.unique(y_true):
        true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
        false_positives = np.sum((y_true != class_label) & (y_pred == class_label))
        false_negatives = np.sum((y_true == class_label) & (y_pred != class_label))

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    return macro_f1


def calculate_micro_f1_score(y_true, y_pred):
    true_positives = np.sum((y_true == y_pred) & (y_pred == 1))
    false_positives = np.sum((y_true != y_pred) & (y_pred == 1))
    false_negatives = np.sum((y_true != y_pred) & (y_pred == 0))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    micro_f1 = 2 * (precision * recall) / (precision + recall)

    return micro_f1


def generate_fake_flattened_images(num_images, image_size=784, num_classes=10):
    images = []
    labels = []

    for _ in range(num_images):
        label = np.random.randint(num_classes)
        labels.append(label)

        image = np.random.randint(0, 256, size=image_size, dtype=np.uint8)
        images.append(image)

    return np.array(images), np.array(labels)


def one_hot_encode_list(label_list):
    unique_classes = len(np.unique(label_list))
    num_samples = len(label_list)
    one_hot_labels = np.zeros((num_samples, unique_classes))

    for i, label in enumerate(label_list):
        one_hot_labels[i, label] = 1

    return one_hot_labels.tolist()


def train_test_split(data, targets, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = len(data)
    num_test_samples = int(test_size * num_samples)

    test_indices = np.random.choice(num_samples, num_test_samples, replace=False)

    X_train = [data[i] for i in range(num_samples) if i not in test_indices]
    X_test = [data[i] for i in test_indices]
    y_train = [targets[i] for i in range(num_samples) if i not in test_indices]
    y_test = [targets[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def plot_performance(train_accuracies, train_losses, val_accuracies, val_losses, save_name):
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)

    # Plot train accuracy and validation accuracy
    plt.plot(train_accuracies, label='Train Accuracy', marker='o', linestyle='-')
    plt.plot(val_accuracies, label='Validation Accuracy', marker='x', linestyle='-')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracies Over Epochs')
    plt.legend()

    # Customize grid lines
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.subplot(1, 2, 2)

    # Plot train loss and validation loss
    plt.plot(train_losses, label='Train Loss', marker='o', linestyle='-')
    plt.plot(val_losses, label='Validation Loss', marker='x', linestyle='-')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses Over Epochs')
    plt.legend()

    # Customize grid lines
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"Results/{save_name}.png")

def plot_wrong_predictions(wrong_predictions, num_examples_per_row=3):
    num_wrong = len(wrong_predictions)
    num_rows = (num_wrong + num_examples_per_row - 1) // num_examples_per_row

    fig, axes = plt.subplots(num_rows, num_examples_per_row, figsize=(12, 4 * num_rows))

    for i, (image, predicted, true_label) in enumerate(wrong_predictions):
        image = np.reshape(image, (28, 28))

        row = i // num_examples_per_row
        col = i % num_examples_per_row
        ax = axes[row, col]

        # Display the image
        ax.imshow(image, cmap='gray')
        ax.axis('off')

        # Set the title with predicted and true labels
        ax.set_title(f'Predicted:{predicted}, Label:{true_label}')

    # Remove empty subplots
    for i in range(len(wrong_predictions), num_rows * num_examples_per_row):
        row = i // num_examples_per_row
        col = i % num_examples_per_row
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()


def plot_class_errors(wrong_predictions, test_labels):
    # Extract true labels from wrong predictions
    true_labels = [true_label for _, _, true_label in wrong_predictions]

    # Calculate the number of each label in the test set
    class_counts = np.bincount(test_labels)

    # Initialize ratios of incorrect predictions to total labels for each class
    class_error_ratios = np.zeros(len(class_counts))

    # Calculate the ratios for each class
    for true_label, predicted_label in zip(true_labels, test_labels):
        if true_label != predicted_label:
            class_error_ratios[true_label] += 1

    class_error_ratios /= class_counts  # Calculate the ratios
    class_error_percentages = class_error_ratios * 100  # Convert to percentages

    # Set the ggplot style
    plt.style.use('ggplot')

    # Create a bar plot for the percentages
    classes = np.arange(len(class_counts))
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(classes, class_error_percentages, color='#FF6B6B', alpha=0.7)

    ax.set_xlabel("Class")
    ax.set_ylabel("Incorrect Predictions (%)")
    ax.set_title("Percentage of Incorrect Predictions by Class")
    ax.set_xticks(classes)

    plt.show()