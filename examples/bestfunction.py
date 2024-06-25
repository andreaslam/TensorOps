import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from node import Node, NodeContext, forward, backward, zero_grad
from loss import L1Loss
from utils import LossPlotter, visualise_graph
from optim import Adam
from model import Model, sigmoid
import random

class LinearApproximator(Model):  # fits using y = m * x + c
    def __init__(self, weights):
        super().__init__(weights)
        self.name = "linear-approx"
        self.m = weights[0]
        self.c = weights[1]

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
        X_train = [Node(random.uniform(-1,1), requires_grad=False, weight=False) for _ in range(10)]
        y_train = [
            Node(
                random.uniform(0, 1),
                requires_grad=False,
                weight=False,
            )
            for _ in X_train
        ]

        linear = LinearApproximator(
            [
                Node(random.uniform(-1, 1), requires_grad=True, weight=True)
                for _ in range(2)
            ]
        )
        quadratic = QuadraticApproximator(
            [
                Node(random.uniform(-1, 1), requires_grad=True, weight=True)
                for _ in range(3)
            ]
        )
        cubic = CubicApproximator(
            [
                Node(random.uniform(-1, 1), requires_grad=True, weight=True)
                for _ in range(4)
            ]
        )
        # Perform forward pass
        
        print([(x.value, y.value) for x, y in zip(X_train, y_train)])
        print("linear model containing weights:", linear)
        print("quadratic model containing weights:", quadratic)
        print("cubic model containing weights:", cubic)

        approximators = [linear, quadratic, cubic]

        loss_fn = L1Loss()
        loss_plot = LossPlotter()
        for approximator in approximators:
            optim = Adam(approximator.weights, lr=1e-5)
            total_loss_per_approx = []
            for epoch in range(10):
                for X, y in zip(X_train, y_train):
                    zero_grad(context.nodes)
                    y_preds = approximator.forward(X)
                    forward(context.nodes)
                    print("y_preds", y_preds)
                    print("y", y)
                    loss = loss_fn.loss(y_preds.value, y.value)
                    print(loss)
                    backward(context.nodes)
                    optim.step()
                    total_loss_per_approx.append(loss)
                    loss_plot.register_datapoint(loss, label=approximator.name)
                results[approximator.name] = sum(total_loss_per_approx) / len(
                    total_loss_per_approx
                )
        loss_plot.plot()
        
        print("linear model containing weights:", linear)
        print("quadratic model containing weights:", quadratic)
        print("cubic model containing weights:", cubic)
        
        print("Loss statistics (average):")
        for approx in approximators:
            print(f"{approx.name}: {results[approx.name]:.6f}")
        print(
            f"Best function: {min(results, key=results.get)} with average loss of {min(results.values()):.6f}"
        )