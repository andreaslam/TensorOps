import torch
import torch.nn as nn
import random


def init_network_params(layer):
    layer.weight = nn.Parameter(
        torch.tensor(
            [
                [random.uniform(-1, 1) for _ in range(layer.in_features)]
                for _ in range(layer.out_features)
            ],
            dtype=torch.float64,
        )
    )
    layer.bias = nn.Parameter(
        torch.tensor(
            [random.uniform(-1, 1) for _ in range(layer.out_features)],
            dtype=torch.float64,
        )
    )
