# creates 3 different polynomials to fit a dataset and finds the "best" function by comparing and plotting their MSE losses

from node import Node, NodeContext, forward, backward, zero_grad
from loss import MSELoss
from utils import LossPlotter
from optim import SGD
from model import Model
import random


class LinearApproximator(Model):  # # fits using y = m * x + c
    def __init__(self, weights):
        super().__init__(weights)
        self.name = "linear-approx"
        self.m = weights[0]
        self.c = weights[1]

    def forward(self, x):
        with NodeContext() as cxt:
            cxt.nodes += self.weights
            cxt.nodes.append(x)
            x = self.m * x + self.c
            forward(cxt.nodes)
            return x


class QuadraticApproximator(Model):  # fits using y = ax^2 + bx + c
    def __init__(self, weights):
        super().__init__(weights)
        self.name = "quad-approx"
        self.a = weights[0]
        self.b = weights[1]
        self.c = weights[2]

    def forward(self, x):
        with NodeContext() as cxt:
            cxt.nodes += self.weights
            cxt.nodes.append(x)
            x = self.a * (x**2) + self.b * x + self.c
            forward(cxt.nodes)
            return x


class CubicApproximator(Model):  # fits using y = ax^3 + bx^2 + cx + d
    def __init__(self, weights):
        super().__init__(weights)
        self.name = "cubic-approx"
        self.a = weights[0]
        self.b = weights[1]
        self.c = weights[2]
        self.d = weights[3]

    def forward(self, x):
        with NodeContext() as cxt:
            cxt.nodes += self.weights
            cxt.nodes.append(x)
            x = self.a * (x**3) + self.b * (x**2) + self.c * x + self.d
            forward(cxt.nodes)
            return x


if __name__ == "__main__":
    random.seed(42)
    results = {}
    with NodeContext() as context:
        X_train = [Node(random.uniform(-100, 100)) for _ in range(0, 100)]
        y_train = [Node(x.value+random.uniform(-5, 5)) for x in X_train]

        print([(x.value, y.value) for x, y in zip(X_train, y_train)])

        linear = LinearApproximator(
            [Node(random.uniform(-3, 3), requires_grad=True) for _ in range(2)]
        )
        quadratic = QuadraticApproximator(
            [Node(random.uniform(-2, 2), requires_grad=True) for _ in range(3)]
        )
        cubic = CubicApproximator(
            [Node(random.uniform(-1, 1), requires_grad=True) for _ in range(4)]
        )

        print("linear model containing weights:", linear)
        print("quadratic model containing weights:", quadratic)
        print("cubic model containing weights:", cubic)

        approximators = [linear]

        loss_fn = MSELoss()
        loss_plot = LossPlotter()

        for approximator in approximators:
            optim = SGD(approximator.weights, lr=1e-6)
            total_loss_per_approx = []
            for epoch in range(10):
                for X, y in zip(X_train, y_train):
                    zero_grad(approximator.weights)
                    backward(approximator.weights)
                    y_preds = approximator.forward(X)
                    loss = loss_fn.loss(y_preds.value, y.value)
                    # print(f"{approximator.name} epoch: {epoch} loss: {loss:.3f}")
                    loss_plot.register_datapoint(loss, label=approximator.name)
                    total_loss_per_approx.append(loss)
                    optim.step()
                results[approximator.name] = sum(total_loss_per_approx) / len(
                    total_loss_per_approx
                )
            print(approximator.weights)
        loss_plot.plot()

        print(
            f"Best function: {min(results,key=results.get)} with average loss of {min(results.values()):.3f}"
        )
