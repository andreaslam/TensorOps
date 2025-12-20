# Train a neural network on the MNIST dataset to recognise handwritten digits.
# This code is to be used as comparison with examples/pytorch/mnist.py

import gzip
import os
import random
import struct
from urllib.request import urlretrieve

from tensorops.loss import CrossEntropyLoss
from tensorops.optim import AdamW
from tensorops.tensor import Tanh, Tensor, TensorContext
from tensorops.utils.models import SequentialModel
from tensorops.utils.tensorutils import PlotterUtil

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
            # Normalize to mean 0, std 1 (approx)
            image = [(x / 255.0 - 0.1307) / 0.3081 for x in image]
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


(train_images, train_labels), (test_images, test_labels) = load_mnist()


class MNISTModel(SequentialModel):
    def __init__(
        self,
        num_hidden_layers: int,
        num_hidden_nodes: int,
        loss_criterion,
        seed: int | None = None,
        activation_function=Tanh,
        *,
        batch_size: int = 1,
    ) -> None:
        super().__init__(loss_criterion, seed, batch_size=batch_size)
        self.activation_function = activation_function
        self.num_hidden_layers = num_hidden_layers
        with self.context:
            self.add_layer(784, num_hidden_nodes, self.activation_function)
            for _ in range(self.num_hidden_layers):
                self.add_layer(
                    num_hidden_nodes, num_hidden_nodes, self.activation_function
                )
            # Final layer emits logits; softmax is handled inside CrossEntropyLoss.
            self.add_layer(num_hidden_nodes, 10, None)
            # CrossEntropyLoss expects (logits, target).
            self.loss = self.loss_criterion(
                self.model_output_layer.layer_output, self.targets
            )

    def forward(self, model_inputs: Tensor) -> Tensor:  # type: ignore[override]
        with self.context:
            # Input must be (batch_size, 784) for this model.
            for layer in self.model_layers:
                layer.forward(model_inputs)
                model_inputs = layer.layer_output
            return model_inputs


with TensorContext() as tc:
    X_train, y_train, X_test, y_test = (
        Tensor(train_images, requires_grad=False),
        Tensor(train_labels, requires_grad=False),
        Tensor(test_images, requires_grad=False),
        Tensor(test_labels, requires_grad=False),
    )

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # impl model
    BATCH_SIZE = 32
    N_EPOCHS = 100

    model = MNISTModel(
        1,
        128,
        CrossEntropyLoss(),
        seed=42,
        batch_size=BATCH_SIZE,
        activation_function=Tanh,
    )
    optim = AdamW(model.get_weights(), lr=1e-4, weight_decay=1e-4)
    # Enable gradient clipping to stabilise updates
    optim.grad_clip_norm = 0.5
    optim.grad_clip_value = 0.5
    model.train()

    loss_plot = PlotterUtil()

    # Helper to create fixed-size batches (graph uses a fixed batch_size)
    def _one_hot(label: int, num_classes: int = 10) -> list[float]:
        vec = [0.0] * num_classes
        vec[int(label)] = 1.0
        return vec

    def _has_nonfinite_grad(params: list[Tensor]) -> bool:
        import numpy as np

        for p in params:
            g = getattr(p, "grads", None)
            if g is None:
                continue
            src = g.flat if getattr(g, "flat", None) is not None else g.values
            if src is None:
                continue
            arr = np.array(src, dtype=float)
            if not np.isfinite(arr).all():
                return True
        return False

    def get_batches(images: Tensor, labels: Tensor, batch_size: int):
        """Yield full (batch_size, 784) images and (batch_size, 10) one-hot labels."""
        assert images.values is not None and labels.values is not None
        n_samples = len(images.values)
        indices = list(range(n_samples))
        random.shuffle(indices)

        # Drop the last partial batch to keep shapes constant.
        for start_idx in range(0, n_samples - batch_size + 1, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch_images = [images.values[i] for i in batch_indices]
            batch_labels = [_one_hot(int(labels.values[i])) for i in batch_indices]
            yield (
                Tensor(batch_images, requires_grad=False),
                Tensor(batch_labels, requires_grad=False),
            )

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in get_batches(X_train, y_train, BATCH_SIZE):
            model.zero_grad()

            # Wire inputs/targets into the existing graph without creating new ops
            logits = model(X_batch, execute=False)
            assert model.targets is not None
            model.targets.values = y_batch.values

            # Re-run the fixed graph with updated data
            model.context.forward(recompute=True)

            # Debug logits and guard against non-finite values
            vals = logits.flat
            import numpy as np

            v = np.array(vals)
            print(f"Logits: min={v.min()}, max={v.max()}, mean={v.mean()}")
            if not np.isfinite(v).all():
                print("Skipping batch due to non-finite logits")
                continue

            # Loss is part of the pre-built graph; reuse it to avoid graph bloat
            loss = model.loss

            model.backward()

            # Skip update if loss blew up or produced bad grads
            loss_val = loss.item()
            if not np.isfinite(loss_val):
                print("Skipping batch due to non-finite loss")
                model.zero_grad()
                continue

            if _has_nonfinite_grad(model.get_weights()):
                print("Skipping batch due to non-finite gradients")
                model.zero_grad()
                continue

            optim.step()

            # Clip weights in-place to keep logits bounded
            for p in model.get_weights():
                src = p.flat if getattr(p, "flat", None) is not None else p.values
                if src is None:
                    continue
                # Handle NaNs in weights by resetting to 0.0 or small random values
                # (If weights are NaN, the model is broken, but we try to recover)
                clipped = []
                for x in src:
                    val = float(x)
                    if not np.isfinite(val):
                        clipped.append(0.0)
                    else:
                        clipped.append(max(-0.5, min(0.5, val)))
                p.values = clipped

            batch_loss = loss_val
            epoch_loss += batch_loss
            print(f"Batch loss: {batch_loss:.4f}")
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_plot.register_datapoint(avg_loss, "MNIST-TensorOps")
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")
