# Given three polynomial fitters constructed using the torch.nn.Module base class and fits to the point (2,1).
# The code will train each model for 100 "epochs" and return the best performing fitter with its respective loss.
# This code is to be used as comparison with bestfunction.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Parameter(torch.tensor(0.7))
        self.c = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        return self.m * x + self.c


class QuadraticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.7))
        self.b = nn.Parameter(torch.tensor(0.3))
        self.c = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        return self.a * (x**2) + self.b * x + self.c


class CubicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.7))
        self.b = nn.Parameter(torch.tensor(0.3))
        self.c = nn.Parameter(torch.tensor(0.3))
        self.d = nn.Parameter(torch.tensor(-0.7))

    def forward(self, x):
        return self.a * (x**3) + self.b * (x**2) + self.c * x + self.d


class LossPlotter:
    def __init__(self):
        self.losses = {}

    def register_datapoint(self, loss, model_name):
        if model_name not in self.losses:
            self.losses[model_name] = []
        self.losses[model_name].append(loss)

    def plot(self):
        for model_name, losses in self.losses.items():
            plt.plot(losses, label=model_name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


def train_model(model, criterion, optimiser, num_iterations, loss_plot, results):
    for _ in tqdm(
        range(num_iterations), desc=f"Training {type(model).__name__}-TensorOps"
    ):
        optimiser.zero_grad()
        output = model(torch.tensor(2.0))
        loss = criterion(output, torch.tensor(1.0))
        loss.backward()
        optimiser.step()
        loss_plot.register_datapoint(loss.item(), f"{type(model).__name__}-PyTorch")
    results[f"{type(model).__name__}-PyTorch"] = loss.item()
    return results


if __name__ == "__main__":
    models = [LinearModel(), QuadraticModel(), CubicModel()]
    criterion = nn.MSELoss()
    loss_plot = LossPlotter()
    results = {}
    for model in models:
        optimiser = optim.Adam(model.parameters(), lr=5e-3)
        train_model(model, criterion, optimiser, 100, loss_plot, results)
    print("Training results:", results)
    print(
        f"Best model was {min(results, key=results.get)} with loss {results[min(results, key=results.get)]}"
    )
    loss_plot.plot()
