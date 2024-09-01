# creates a fully connected multi-layer perceptron (MLP) using `torch.nn.Module`

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorops.utils.tensorutils import PlotterUtil
import random
from helpers import init_network_params


class MLP(nn.Module):
    def __init__(
        self,
        num_input_nodes,
        num_output_nodes,
        num_hidden_layers,
        num_hidden_nodes,
        activation_function=nn.Sigmoid(),
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.num_hidden_layers = num_hidden_layers
        input_layer = nn.Linear(num_input_nodes, num_hidden_nodes)
        init_network_params(input_layer)
        self.layers.append(input_layer)
        for _ in range(self.num_hidden_layers):
            hidden_layer = nn.Linear(num_hidden_nodes, num_hidden_nodes)
            init_network_params(hidden_layer)
            self.layers.append(hidden_layer)
        output_layer = nn.Linear(num_hidden_nodes, num_output_nodes)
        init_network_params(output_layer)
        self.layers.append(output_layer)
        self.activation_function = activation_function

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_function(layer(x))
        return x


def prepare_training_inputs():
    X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train, X_test, y_train, y_test


def run_training_loop(
    X_train, y_train, model, loss_criterion, optimiser, loss_plot, num_epochs
):
    for _ in tqdm(range(num_epochs)):
        model.train()
        optimiser.zero_grad()
        outputs = model(X_train)
        loss = loss_criterion(outputs, y_train)
        loss_plot.register_datapoint(loss.item(), f"{type(model).__name__}-PyTorch")
        loss.backward()
        optimiser.step()

    print(f"Train Loss: {loss.item():.5f}")


if __name__ == "__main__":
    random.seed(42)

    num_epochs = 100

    num_input_nodes = 2
    num_output_nodes = 1
    num_hidden_nodes = 30
    num_hidden_layers = 30

    # ok init hyperparams

    # num_input_nodes = 2
    # num_output_nodes = 1
    # num_hidden_nodes = 30
    # num_hidden_layers = 30

    X_train, X_test, y_train, y_test = prepare_training_inputs()
    model = MLP(num_input_nodes, num_output_nodes, num_hidden_layers, num_hidden_nodes)

    loss_criterion = nn.MSELoss()

    optimiser = optim.SGD(model.parameters(), lr=1e-1)
    loss_plot = PlotterUtil()

    for layer in model.layers:
        print("Layer weights:", layer.weight)
        print("Layer biases:", layer.bias)

    run_training_loop(
        X_train, y_train, model, loss_criterion, optimiser, loss_plot, num_epochs
    )

    model.eval()

    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = loss_criterion(y_pred, y_test)
        print(f"Test Loss: {test_loss.item():.5f}")

    loss_plot.plot()
