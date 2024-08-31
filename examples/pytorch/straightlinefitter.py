# Given a model y = mx + c, constructed using the torch.nn.Module base class and fits to the equation y = 1
# The code will train each model for 100 "epochs" and return the best performing fitter with its respective loss.
# This code is to be used as comparison with examples/tensorops/straightlinefitter.py


import random
import torch
import torch.nn as nn
import torch.optim as optim
from tensorops.utils.tensorutils import PlotterUtil
from tqdm import tqdm


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.m = nn.Parameter(torch.tensor(0.7, requires_grad=True))
        self.c = nn.Parameter(torch.tensor(0.3, requires_grad=True))

    def forward(self, x):
        return self.m * x + self.c


def train_model(model, optimiser, num_iterations, loss_plot):
    criterion = nn.MSELoss()
    random.seed(42)
    for _ in tqdm(
        range(num_iterations), desc=f"Training {type(model).__name__}-PyTorch"
    ):
        model.train()
        optimiser.zero_grad()
        input_value = torch.tensor(random.randint(-5, 5), requires_grad=False)
        target_value = torch.tensor(1.0, requires_grad=False)
        output = model(input_value)
        loss = criterion(output, target_value)
        loss.backward()
        optimiser.step()
        loss_plot.register_datapoint(loss.item(), f"{type(model).__name__}-PyTorch")
    loss_plot.plot()
    print(list(model.parameters()))


if __name__ == "__main__":
    model = LinearModel()
    optimiser = optim.Adam(model.parameters(), lr=5e-2)
    loss_plot = PlotterUtil()
    train_model(model, optimiser, 100, loss_plot)
