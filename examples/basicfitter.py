# Fits a polynomial (ax^2 + bx + c) to a single value and optimised using SGD() and using tensorops.loss.MSELoss as cost function
# This example does not use the tensorops.Model class but uses tensorops.Node to construct a polynomial

from tensorops.node import Node, NodeContext, forward, backward, zero_grad
from tensorops.loss import MSELoss
from tensorops.tensorutils import LossPlotter
from tensorops.optim import SGD

if __name__ == "__main__":
    with NodeContext() as context:
        target = Node(10.0, requires_grad=False, weight=False)
        x = Node(0.5, requires_grad=False, weight=False)
        w_0 = Node(0.9, requires_grad=True, weight=True)
        w_1 = Node(0.6, requires_grad=True, weight=True)
        w_2 = Node(0.4, requires_grad=True, weight=True)

        polynomial_result = w_0 + w_1 * x + w_2 * (x**2)  # predictor function
        loss_fn = MSELoss()
        loss = loss_fn.loss(polynomial_result, target)
        loss_plot = LossPlotter()
        optim = SGD([w_0, w_1, w_2], lr=0.1)

        for gen in range(10):
            zero_grad(context.nodes)
            forward(context.nodes)
            backward(context.nodes)

            print(f"generation {gen}: {loss.value:2f}")

            optim.step()
            loss_plot.register_datapoint(loss.value, "ax^2+bx+c (TensorOps)")
        loss_plot.plot()
