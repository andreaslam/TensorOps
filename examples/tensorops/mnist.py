import os
from re import X
import struct
import gzip
from urllib.request import urlretrieve

from tensorops.newtensor import Tensor, TensorContext

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


(train_images, train_labels), (test_images, test_labels) = load_mnist()

with TensorContext() as tc:
    X_train, y_train, X_test, y_test = (
        Tensor(train_images, requires_grad=False),
        Tensor(train_labels, requires_grad=False),
        Tensor(test_images, requires_grad=False),
        Tensor(test_labels, requires_grad=False),
    )

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # impl model
    # impl loading
    # impl batching
    # impl backward
    # impl optim
    # impl device
