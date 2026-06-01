# Train a neural network on the MNIST dataset to recognise handwritten digits.
# This code is to be used as comparison with examples/pytorch/mnist.py

import gzip
import os
import random
import struct
from urllib.request import urlretrieve

# Device optimizer requires DirectInput caching for all params (including biases).
os.environ.setdefault("TENSOROPS_DIRECT_INPUT_CACHE", "1")
os.environ.setdefault("TENSOROPS_DIRECT_INPUT_CACHE_MIN_LEN", "1")

from tensorops.loss import CrossEntropyLoss
from tensorops.optim import AdamWDevice
from tensorops.tensor import LeakyReLU, Tensor
from tensorops.utils.models import SequentialModel

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
        header = f.read(16)
        magic_number, num_images, rows, cols = struct.unpack(">IIII", header)
        assert magic_number == 2051, (
            f"Invalid magic number {magic_number} in image file."
        )

        images = []
        for _ in range(num_images):
            image = list(f.read(rows * cols))
            images.append(image)
        return images


def extract_labels(filepath):
    """Extract labels from the .gz file and return them as a list."""
    with gzip.open(filepath, "rb") as f:
        header = f.read(8)
        magic_number, num_labels = struct.unpack(">II", header)
        assert magic_number == 2049, (
            f"Invalid magic number {magic_number} in label file."
        )

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


def _normalise_images(images):
    return [[x / 255.0 for x in image] for image in images]


def init_layer_params(layer, rng):
    """Match the PyTorch init_network_params helper (uniform -1..1)."""
    weights_out_in = [
        [rng.uniform(-1.0, 1.0) for _ in range(layer.num_input_tensors)]
        for _ in range(layer.num_output_tensors)
    ]
    # TensorOps expects (in, out) weights for x @ W.
    weights_in_out = [list(col) for col in zip(*weights_out_in)]
    layer.output_weights.values = weights_in_out
    layer.output_bias.values = [
        [rng.uniform(-1.0, 1.0) for _ in range(layer.num_output_tensors)]
    ]


def init_model_params(model, seed=42):
    rng = random.Random(seed)
    for layer in model.model_layers:
        init_layer_params(layer, rng)


class MNISTModel(SequentialModel):
    def __init__(
        self,
        num_hidden_layers: int,
        num_hidden_nodes: int,
        loss_criterion,
        activation_function=LeakyReLU,
        *,
        batch_size: int = 1,
    ) -> None:
        super().__init__(loss_criterion, None, batch_size=batch_size)
        self.activation_function = activation_function
        self.num_hidden_layers = num_hidden_layers
        with self.context:
            self.add_layer(784, num_hidden_nodes, self.activation_function)
            for _ in range(self.num_hidden_layers):
                self.add_layer(
                    num_hidden_nodes, num_hidden_nodes, self.activation_function
                )
            # Apply activation on the output layer to match the PyTorch example.
            self.add_layer(num_hidden_nodes, 10, self.activation_function)
            # CrossEntropyLoss expects (logits, target).
            self.loss = self.loss_criterion(
                self.model_output_layer.layer_output, self.targets
            )

    def forward(self, model_inputs: Tensor) -> Tensor:  # type: ignore[override]
        with self.context:
            # Update only the input placeholder; the graph is already wired.
            if self.model_input_layer is None or self.model_output_layer is None:
                raise ValueError("Model layers are not initialised")
            if isinstance(model_inputs, Tensor):
                self.model_input_layer.layer_input_tensors.values = model_inputs.values
            else:
                self.model_input_layer.layer_input_tensors.values = model_inputs
            return self.model_output_layer.layer_output


if __name__ == "__main__":
    random.seed(42)
    (train_images, train_labels), (test_images, test_labels) = load_mnist()

    train_images = _normalise_images(train_images)
    test_images = _normalise_images(test_images)

    X_train, y_train, X_test, y_test = (
        train_images,
        train_labels,
        test_images,
        test_labels,
    )

    BATCH_SIZE = 256
    N_EPOCHS = 100

    model = MNISTModel(
        2,
        256,
        CrossEntropyLoss(),
        batch_size=BATCH_SIZE,
        activation_function=LeakyReLU,
    )
    init_model_params(model, seed=42)

    model.train()
    optim = AdamWDevice(model.get_weights(), lr=2e-4)

    # Helper to create fixed-size batches (graph uses a fixed batch_size)
    def _one_hot_labels(labels: list[int], num_classes: int = 10) -> list[list[float]]:
        one_hot = [[0.0] * num_classes for _ in labels]
        for i, lbl in enumerate(labels):
            one_hot[i][int(lbl)] = 1.0
        return one_hot

    y_train_one_hot = _one_hot_labels(y_train)
    y_test_one_hot = _one_hot_labels(y_test)

    def get_batches(images, labels_one_hot, batch_size: int, *, shuffle=True):
        """Yield full (batch_size, 784) images and (batch_size, 10) one-hot labels."""
        n_samples = len(images)
        indices = list(range(n_samples))
        if shuffle:
            random.shuffle(indices)

        # Drop the last partial batch to keep shapes constant.
        for start_idx in range(0, n_samples - batch_size + 1, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch_images = [images[i] for i in batch_indices]
            batch_labels = [labels_one_hot[i] for i in batch_indices]
            yield batch_images, batch_labels

    def get_eval_batches(images, labels_one_hot, batch_size: int):
        """Yield eval batches, padding the last batch to keep shapes constant."""
        n_samples = len(images)

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = list(
                range(start_idx, min(start_idx + batch_size, n_samples))
            )
            valid_count = len(batch_indices)
            if valid_count < batch_size:
                batch_indices.extend([batch_indices[-1]] * (batch_size - valid_count))

            batch_images = [images[i] for i in batch_indices]
            batch_labels = [labels_one_hot[i] for i in batch_indices]
            yield batch_images, batch_labels, valid_count

    # Reuse model input/target tensors to avoid per-batch Tensor allocations.
    input_tensor = model.model_input_layer.layer_input_tensors
    assert model.targets is not None
    target_tensor = model.targets

    for epoch in range(N_EPOCHS):
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}")

        for id_batch, (batch_images, batch_labels) in enumerate(
            get_batches(X_train, y_train_one_hot, BATCH_SIZE, shuffle=True)
        ):
            model.zero_grad()

            input_tensor.values = batch_images
            target_tensor.values = batch_labels

            model.context.forward(recompute=True)
            loss = model.loss
            model.backward(device_optim=optim)

            if id_batch % 250 == 0:
                loss_value = loss.item()
                print(f"Loss: {loss_value:.4f}")

    model.eval()
    correct = 0
    total = 0
    import numpy as np

    for batch_images, batch_labels, valid_count in get_eval_batches(
        X_test, y_test_one_hot, BATCH_SIZE
    ):
        input_tensor.values = batch_images
        model.context.forward(recompute=True)
        logits = model.model_output_layer.layer_output

        vals = np.array(logits.flat).reshape((BATCH_SIZE, 10))
        predicted = np.argmax(vals, axis=1)[:valid_count]

        y_vals = np.array(batch_labels).reshape((BATCH_SIZE, 10))
        target = np.argmax(y_vals, axis=1)[:valid_count]

        correct += np.sum(predicted == target)
        total += valid_count

    print(f"Test Accuracy: {correct / total:.4f}")
