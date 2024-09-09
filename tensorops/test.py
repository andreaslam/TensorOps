import random
import torch
from torch.optim import Adam
from torch import nn
from tensorutils import LossPlotter
from sklearn.datasets import make_moons


class LinearApproximator(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.name = "linear-approx-pytorch"
        self.m = nn.Parameter(torch.tensor(weights[0]))
        self.c = nn.Parameter(torch.tensor(weights[1]))

    def forward(self, x, n=10):
        if n == 0:
            return x
        else:
            return self.forward(torch.sigmoid(self.m * x + self.c), n - 1)


class QuadraticApproximator(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.name = "quad-approx-pytorch"
        self.a = nn.Parameter(torch.tensor(weights[0]))
        self.b = nn.Parameter(torch.tensor(weights[1]))
        self.c = nn.Parameter(torch.tensor(weights[2]))

    def forward(self, x, n=10):
        if n == 0:
            return x
        else:
            return self.forward(
                torch.sigmoid(self.a * (x**2) + self.b * x + self.c), n - 1
            )


class CubicApproximator(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.name = "cubic-approx-pytorch"
        self.a = nn.Parameter(torch.tensor(weights[0]))
        self.b = nn.Parameter(torch.tensor(weights[1]))
        self.c = nn.Parameter(torch.tensor(weights[2]))
        self.d = nn.Parameter(torch.tensor(weights[3]))

    def forward(self, x, n=10):
        if n == 0:
            return x
        else:
            return self.forward(
                torch.sigmoid(
                    self.a * (x**3) + self.b * (x**2) + self.c * x + self.d
                ),
                n - 1,
            )


if __name__ == "__main__":
    random.seed(42)
    results = {}
    X = [random.uniform(-10, 10) for _ in range(10)]
    y = [random.uniform(0, 1) for _ in X]
    print([(xv, yv) for xv, yv in zip(X, y)])
    # X,y = make_moons(20, noise=0.2, random_state=42)
    # X = list(X.reshape(-1))
    # y = list(y.reshape(-1))
    X = [float(x_data) for x_data in X]
    y = [float(y_data) for y_data in y]
    X_train = torch.tensor(X)
    y_train = torch.tensor(y)

    linear = LinearApproximator([random.uniform(-1, 1) for _ in range(2)])
    quadratic = QuadraticApproximator([random.uniform(-1, 1) for _ in range(3)])
    cubic = CubicApproximator([random.uniform(-1, 1) for _ in range(4)])

    print("linear model containing weights:", list(linear.parameters()))
    print("quadratic model containing weights:", list(quadratic.parameters()))
    print("cubic model containing weights:", list(cubic.parameters()))

    approximators = [linear, quadratic, cubic]

    loss_fn = nn.L1Loss()

    loss_plot = LossPlotter()

    for approximator in approximators:
        total_loss_per_approx = []
        optim = Adam(approximator.parameters(), lr=1e-3)
        for epoch in range(10):
            for X, y in zip(X_train, y_train):
                optim.zero_grad()
                y_pred = approximator(X.float().unsqueeze(0))
                print("y", y)
                print("y_pred", y_pred)
                loss = loss_fn(y_pred, y.float().unsqueeze(0))
                loss.backward()
                optim.step()
                total_loss_per_approx.append(loss.item())
                loss_plot.register_datapoint(loss.detach().numpy(), approximator.name)

        results[approximator.name] = sum(total_loss_per_approx) / len(
            total_loss_per_approx
        )

    loss_plot.plot()

    print("linear model containing weights:", list(linear.parameters()))
    print("quadratic model containing weights:", list(quadratic.parameters()))
    print("cubic model containing weights:", list(cubic.parameters()))

    print("Loss statistics (average):")
    for approx in approximators:
        print(f"{approx.name}: {results[approx.name]:.6f}")
    print(
        f"Best function: {min(results, key=results.get)} with average loss of {min(results.values()):.6f}"
    )
