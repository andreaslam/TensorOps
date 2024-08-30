import torch
import torch.nn as nn
import random


class MLP(nn.Module):
    def __init__(self, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            layer = nn.Linear(10, 10)

            layer.weight.data = torch.tensor(
                [[random.uniform(-1, 1) for _ in range(10)] for _ in range(10)],
                dtype=torch.float32,
            )
            layer.bias.data = torch.tensor(
                [random.uniform(-1, 1) for _ in range(10)], dtype=torch.float32
            )

            self.layers.append(layer)

        self.sigmoid = nn.Sigmoid()
        self.loss_criterion = nn.MSELoss()

    def forward(self, x):
        for layer in self.layers:
            x = self.sigmoid(layer(x))
        return x

    def calculate_loss(self, output, target):
        return self.loss_criterion(output, target)


if __name__ == "__main__":
    random.seed(42)

    model = MLP(num_layers=3)

    X = torch.tensor([random.uniform(-1, 1) for _ in range(10)], dtype=torch.float32)

    y = model(X)

    print(y)
