# Build and train a neural network layer using `torch.nn.Module`
# This code is to be used as comparison with examples/tensorops/layer.py


import torch
import torch.nn as nn
import torch.optim as optim
import random
from helpers import init_network_params
from tensorops.utils.tensorutils import PlotterUtil


class LayerTest(nn.Module):
    def __init__(self, input_size, output_size, activation_fn):
        super(LayerTest, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation_function = activation_fn

    def forward(self, x):
        x = self.linear(x)
        return self.activation_function(x)


if __name__ == "__main__":
    random.seed(42)

    num_input_nodes = 3
    num_output_nodes = 3
    num_epochs = 10

    num_data = 10

    X_train = [
        torch.tensor([random.uniform(-0.1, 0.1) for _ in range(num_input_nodes)])
        for _ in range(num_data)
    ]
    y_train = [
        torch.tensor([random.uniform(-0.1, 0.1) for _ in range(num_output_nodes)])
        for _ in range(num_data)
    ]

    layer = LayerTest(num_input_nodes, num_output_nodes, torch.sigmoid)

    init_network_params(layer.linear)

    loss_criterion = nn.MSELoss()

    loss_plot = PlotterUtil()

    optimiser = optim.Adam(layer.parameters(), lr=7e-2)

    for _ in range(num_epochs):
        for X, y in zip(X_train, y_train):
            layer.zero_grad()
            y_preds = layer(X)
            loss = loss_criterion(y_preds, y)
            loss.backward()
            optimiser.step()
            loss_plot.register_datapoint(loss.item(), f"{type(layer).__name__}-PyTorch")
    loss_plot.plot()
