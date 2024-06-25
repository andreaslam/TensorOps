import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# fits a polynomial (ax^2 + bx + c) to a single value and optimised using SGD() and using MSELoss as cost function
# this example does not use the Model class but uses Node to construct a polynomial

from node import Node, NodeContext, forward, backward, zero_grad
from loss import MSELoss
from utils import LossPlotter, visualise_graph
from optim import SGD

if __name__ == "__main__":
    with NodeContext() as context:
        target = Node(2.0, requires_grad=False)
        x = Node(0.5, requires_grad=False)
        w_0 = Node(0.9)
        w_1 = Node(0.6)
        w_2 = Node(0.4)

        polynomial_result = w_0 + w_1 * x + w_2 * (x**2)  # predictor function
        loss_fn = MSELoss()
        print(type(polynomial_result), type(target))
        loss = loss_fn.loss(polynomial_result, target)
        loss_plot = LossPlotter()
        optim = SGD([w_0, w_1, w_2], lr=0.1)

        for gen in range(10):
            zero_grad(context.nodes)
            forward(context.nodes)
            backward(context.nodes)
            print(f"generation {gen}: {loss.value:2f}")

            optim.step()
            loss_plot.register_datapoint(loss.value, "ax^2+bx+c")
        loss_plot.plot()

        visualise_graph(context.nodes)
