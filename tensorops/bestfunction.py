from node import Node, NodeContext, forward, backward, zero_grad
from loss import L1Loss
from tensorutils import LossPlotter
from optim import Adam
from model import Model, sigmoid
import random


class LinearApproximator(Model):  # fits using y = m * x + c
    def __init__(self, weights):
        super().__init__(weights)
        self.name = "linear-approx"
        self.m, self.c = weights

    def forward(self, x):
        return sigmoid(self.m * x + self.c)


class QuadraticApproximator(Model):  # fits using y = ax^2 + bx + c
    def __init__(self, weights):
        super().__init__(weights)
        self.name = "quad-approx"
        self.a = weights[0]
        self.b = weights[1]
        self.c = weights[2]

    def forward(self, x):
        return sigmoid(self.a * (x**2) + self.b * x + self.c)


class CubicApproximator(Model):  # fits using y = ax^3 + bx^2 + cx + d
    def __init__(self, weights):
        super().__init__(weights)
        self.name = "cubic-approx"
        self.a = weights[0]
        self.b = weights[1]
        self.c = weights[2]
        self.d = weights[3]

    def forward(self, x):
        return sigmoid(self.a * (x**3) + self.b * (x**2) + self.c * x + self.d)


if __name__ == "__main__":
    random.seed(42)
    results = {}
    with NodeContext() as context:
        X_train = [
            Node(random.uniform(-100, 100), requires_grad=False, weight=False)
            for _ in range(10)
        ]
        y_train = [
            Node(
                random.uniform(-100, 100),
                requires_grad=False,
                weight=False,
            )
            for _ in X_train
        ]

        linear = LinearApproximator(
            [
                Node(random.uniform(-100, 100), requires_grad=True, weight=True)
                for _ in range(2)
            ]
        )
        outputs = [linear.forward(X) for X in X_train]
