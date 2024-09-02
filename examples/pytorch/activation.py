# Build and train an artificial neuron using `torch.nn.Module`
# This code is to be used as comparison with examples/tensorops/activation.py


import torch
import torch.nn as nn
import torch.optim as optim
import random
from tensorops.utils.tensorutils import PlotterUtil


class TestActivation(nn.Module):
    def __init__(self, num_inputs):
        super(TestActivation, self).__init__()
        self.weights = nn.Parameter(
            torch.tensor([random.uniform(-1, 1) for _ in range(num_inputs)])
        )
        self.bias = nn.Parameter(torch.tensor(random.uniform(-1, 1)))

    def forward(self, X):
        weighted_sum = torch.dot(self.weights, X) + self.bias
        return torch.sigmoid(weighted_sum)


if __name__ == "__main__":
    random.seed(42)
    num_inputs = 10
    num_epochs = 10

    num_data = 10

    activation = TestActivation(num_inputs)
    print(f"Weights: {activation.weights.data},\nBias: {activation.bias.data}")

    X_train = [
        torch.tensor([random.uniform(-0.1, 0.1) for _ in range(num_inputs)])
        for _ in range(num_data)
    ]
    y_train = [torch.tensor(0.0) for _ in range(num_data)]

    loss_plot = PlotterUtil()
    loss_criterion = nn.MSELoss()

    optimiser = optim.SGD(activation.parameters(), lr=1e-1)

    for _ in range(num_epochs):
        for X, y in zip(X_train, y_train):
            activation.zero_grad()
            y_preds = activation(X)
            print("Output:", y_preds.item())
            loss = loss_criterion(y_preds, y)
            loss.backward()
            optimiser.step()
            loss_plot.register_datapoint(
                loss.item(), f"{type(activation).__name__}-PyTorch"
            )
    loss_plot.plot()
