# Create and train a fully connected multi-layer perceptron (MLP) using `torch.nn.Module`

# Note that the weights are configured manually using the `init_network_params`, which uses the `random` library to generate seeded weights.
# This is because `torch.manual_seed` works differently than `random.seed()` and for reproducibility for the `tensorops` version the code will be using `random`.
# This code is to be used as comparison with examples/tensorops/mlp.py


from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
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


def training_loop(X, y, mlp, optim, loss_criterion, loss_plot, num_epochs):
    for _ in tqdm(range(num_epochs), desc="Training network"):
        optim.zero_grad()
        outputs = mlp(X)
        loss = loss_criterion(outputs, y)
        loss_plot.register_datapoint(loss.item(), f"{type(mlp).__name__}-PyTorch")
        loss.backward()
        optim.step()


if __name__ == "__main__":
    random.seed(42)

    num_epochs = 100
    num_input_nodes = 2
    num_hidden_nodes = 12
    num_hidden_layers = 3
    num_output_nodes = 2

    X = torch.tensor(
        [[random.uniform(-2, 2) for _ in range(num_input_nodes)]], dtype=torch.float32
    )
    y = torch.tensor(
        [[random.uniform(-2, 2) for _ in range(num_output_nodes)]], dtype=torch.float32
    )
    model = MLP(
        num_input_nodes,
        num_output_nodes,
        num_hidden_layers,
        num_hidden_nodes,
        nn.Sigmoid(),
    )

    loss_criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)

    loss_plot = PlotterUtil()

    training_loop(X, y, model, optimizer, loss_criterion, loss_plot, num_epochs)

    loss_plot.plot()
