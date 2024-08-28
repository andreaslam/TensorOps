# Given an equation of a line (y = mx + c) and random inputs from (-5,5) the linear neural network, built and trained using TensorOps, will try and fit to the training data.
# There is random noise added to the resulting y value of the equation. This is to test the model's ability to adjust its weights for a line of best fit.
# This code is to be used as comparison with noisyfitter.py

import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from tensorops.tensorutils import PlotterUtil


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.m = nn.Parameter(torch.tensor(0.6, requires_grad=True))
        self.c = nn.Parameter(torch.tensor(0.7, requires_grad=True))

    def forward(self, x):
        return self.m * x + self.c


if __name__ == "__main__":
    random.seed(42)

    target_m = 2.0
    target_c = 10.0

    X_train = torch.tensor([random.uniform(-5, 5) for _ in range(10)])
    y_train = torch.tensor(
        [x * target_m + target_c + random.uniform(-1, 1) for x in X_train]
    )

    linear_model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

    loss_plot = PlotterUtil()

    graph_plot = PlotterUtil()

    for x, y in zip(X_train, y_train):
        graph_plot.register_datapoint(
            datapoint=y.item(),
            label="Training Data (PyTorch)",
            x=x.item(),
            plot_style="scatter",
            colour="red",
        )

    for _ in tqdm(range(100), desc=f"Training {type(linear_model).__name__}-PyTorch"):
        for X, y in zip(X_train, y_train):
            optimizer.zero_grad()
            output = linear_model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            loss_plot.register_datapoint(
                loss.item(), f"{type(linear_model).__name__}-PyTorch"
            )

    print(list(linear_model.parameters()))
    loss_plot.plot()

    for x, y in zip(X_train, y_train):
        graph_plot.register_datapoint(
            linear_model(x).item(), x=x.item(), label="y=mx+c (PyTorch)"
        )
    graph_plot.plot()
