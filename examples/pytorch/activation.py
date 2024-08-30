# Build an artificial neuron using `torch.nn.Module`
# This code is to be used as comparison with examples/tensorops/activation.py


import torch
import torch.nn as nn
import random


class ActivationTest(nn.Module):
    def __init__(self, num_inputs):
        super(ActivationTest, self).__init__()
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
    model = ActivationTest(num_inputs)
    print(f"Weights: {model.weights.data},\nBias: {model.bias.data}")

    X = torch.tensor([random.uniform(-10, 10) for _ in range(num_inputs)])

    y = model(X)
    print("Output:", y.item())
