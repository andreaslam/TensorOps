import random

from tqdm import tqdm
from tensorops.loss import MSELoss
from tensorops.model import Model
from tensorops.node import Node, relu, tanh
from tensorops.optim import SGD
from tensorops.utils.tensorutils import PlotterUtil


class MNISTModel(Model):
    def __init__(
        self,
        loss_criterion,
        num_input_nodes,
        num_output_nodes,
        num_hidden_layers,
        num_hidden_nodes,
    ):
        super().__init__(loss_criterion)
        self.num_hidden_layers = num_hidden_layers
        with self.context:
            self.add_layer(num_input_nodes, num_hidden_nodes, tanh)
            for _ in range(self.num_hidden_layers):
                self.add_layer(num_hidden_nodes, num_hidden_nodes, relu)
            self.add_layer(num_hidden_nodes, num_output_nodes, None)
            self.loss = self.loss_criterion(
                self.targets, self.output_layer.layer_output_nodes
            )


def training_loop(X_train, y_train, mlp, optim, loss_plot, num_epochs):
    for _ in tqdm(range(num_epochs), desc="Training MNIST network"):
        for X, y in zip(X_train, y_train):
            mlp.zero_grad()
            y_preds = mlp(X)
            loss = mlp.calculate_loss(y_preds, y)
            print(y_preds, y, loss)
            mlp.backward()
            loss_plot.register_datapoint(loss.value, f"{type(mlp).__name__}-TensorOps")
            optim.step()


def load_data(file_paths):
    all_data = []

    for file in file_paths:
        all_data.append(Node.load(file, limit=784 * 10))
    return all_data


if __name__ == "__main__":
    random.seed(42)

    num_epochs = 100
    num_input_nodes = 784
    num_hidden_nodes = 64
    num_hidden_layers = 10
    num_output_nodes = 10
    folder_path = "/Users/andreas/Desktop/Code/RemoteFolder/TensorOps/examples/data/MNIST/processed_node"
    files = [
        f"{folder_path}/test_data_nodes.pkl",
        f"{folder_path}/test_labels_nodes.pkl",
        f"{folder_path}/train_data_nodes.pkl",
        f"{folder_path}/train_labels_nodes.pkl",
    ]

    all_data = load_data(files)

    X_test, y_test, X_train, y_train = all_data

    X = []
    start = 0
    while True:
        X += [X_train[start : start + 784]]
        start += 784
        if start == len(X_train):
            break

    X_train = X

    loss_plot = PlotterUtil()

    model = MNISTModel(
        MSELoss(),
        num_input_nodes,
        num_output_nodes,
        num_hidden_layers,
        num_hidden_nodes,
    )

    optim = SGD(model.get_weights(), lr=1e-4)

    training_loop(
        X_train,
        y_train,
        model,
        optim,
        loss_plot,
        num_epochs,
    )

    loss_plot.plot()
