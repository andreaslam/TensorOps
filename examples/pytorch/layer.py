# Build a neural network layer using `torch.nn.Module`
# This code is to be used as comparison with examples/tensorops/layer.py


import torch
import torch.nn as nn
import random


class Layer(nn.Module):
    def __init__(self, input_size, output_size, activation_fn):
        super(Layer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation_function = activation_fn

    def forward(self, x):
        x = self.linear(x)
        return self.activation_function(x)


def init_network_params(num_input_nodes, num_output_nodes, layer):
    layer.linear.weight = nn.Parameter(
        torch.tensor(
            [
                [random.uniform(-1, 1) for _ in range(num_input_nodes)]
                for _ in range(num_output_nodes)
            ]
        )
    )
    layer.linear.bias = nn.Parameter(
        torch.tensor([random.uniform(-1, 1) for _ in range(num_output_nodes)])
    )


if __name__ == "__main__":
    random.seed(42)

    num_input_nodes = 3
    num_output_nodes = 3

    layer = Layer(num_input_nodes, num_output_nodes, torch.sigmoid)

    init_network_params(num_input_nodes, num_output_nodes, layer)

    print("Layer weights:", layer.linear.weight)
    print("Layer biases:", layer.linear.bias)

    X = torch.tensor([random.uniform(-1, 1) for _ in range(num_input_nodes)])
    y = layer(X)

    print(f"outputs: {y.tolist()}")
