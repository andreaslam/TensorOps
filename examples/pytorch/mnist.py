# Train a neural network on the MNIST dataset to recognise handwritten digits.

# Note that the weights are configured manually using the `init_network_params`, which uses the `random` library to generate seeded weights.
# This is because `torch.manual_seed` works differently than `random.seed()` and for reproducibility for the `tensorops` version the code will be using `random`.
# This code is to be used as comparison with examples/tensorops/mnist.py


import os
import struct
import gzip
from urllib.request import urlretrieve
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from helpers import init_network_params
import torch.optim as optim

MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = "./mnist_data"


def download_mnist():
    """Download the MNIST dataset from alternative mirrors if not already downloaded."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename, url in MNIST_URLS.items():
        filepath = os.path.join(DATA_DIR, os.path.basename(url))
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urlretrieve(url, filepath)
        else:
            print(f"{filename} already exists.")


def extract_images(filepath):
    """Extract images from the .gz file and return them as a list of lists."""
    with gzip.open(filepath, "rb") as f:
        magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert (
            magic_number == 2051
        ), f"Invalid magic number {magic_number} in image file."

        images = []
        for _ in range(num_images):
            image = list(f.read(rows * cols))
            images.append(image)
        return images


def extract_labels(filepath):
    """Extract labels from the .gz file and return them as a list."""
    with gzip.open(filepath, "rb") as f:
        magic_number, num_labels = struct.unpack(">II", f.read(8))
        assert (
            magic_number == 2049
        ), f"Invalid magic number {magic_number} in label file."

        labels = list(f.read(num_labels))
        return labels


def load_mnist():
    """Download, extract, and load MNIST dataset into lists."""
    download_mnist()

    train_images_path = os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")

    train_images = extract_images(train_images_path)
    train_labels = extract_labels(train_labels_path)
    test_images = extract_images(test_images_path)
    test_labels = extract_labels(test_labels_path)

    return (train_images, train_labels), (test_images, test_labels)


class MNISTModel(nn.Module):
    def __init__(
        self, num_hidden_layers, num_hidden_nodes, activation_function=nn.LeakyReLU()
    ):
        super(MNISTModel, self).__init__()
        self.layers = nn.ModuleList()
        self.num_hidden_layers = num_hidden_layers
        input_layer = nn.Linear(784, num_hidden_nodes)
        init_network_params(input_layer)
        self.layers.append(input_layer)
        for _ in range(self.num_hidden_layers):
            hidden_layer = nn.Linear(num_hidden_nodes, num_hidden_nodes)
            init_network_params(hidden_layer)
            self.layers.append(hidden_layer)
        output_layer = nn.Linear(num_hidden_nodes, 10)
        init_network_params(output_layer)
        self.layers.append(output_layer)
        self.activation_function = activation_function

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_function(layer(x))
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    (train_images, train_labels), (test_images, test_labels) = load_mnist()

    y_train, y_test = (
        torch.tensor(train_labels, dtype=torch.int64, device=device),
        torch.tensor(test_labels, dtype=torch.int64, device=device),
    )

    X_train = torch.tensor(train_images, dtype=torch.float64, device=device) / 255.0
    X_test = torch.tensor(test_images, dtype=torch.float64, device=device) / 255.0
    X_train = X_train.view(-1, 784)
    X_test = X_test.view(-1, 784)

    model = MNISTModel(2, 256).to(device)

    model.train()
    BATCH_SIZE = 256
    N_EPOCHS = 100

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)

    dataset_size = len(dataloader.dataset)

    for epoch in range(N_EPOCHS):
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}")

        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            y_batch_pred = model(x_batch)

            loss = loss_fn(y_batch_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if id_batch % 250 == 0:
                loss_value = loss.item()
                current = id_batch * len(x_batch)
                print(f"Loss: {loss_value:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in DataLoader(
            TensorDataset(X_test, y_test), batch_size=BATCH_SIZE
        ):
            y_batch_pred = model(x_batch)
            predicted = torch.argmax(y_batch_pred, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")
