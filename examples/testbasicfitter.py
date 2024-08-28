# Fits a polynomial (ax^2 + bx + c) to a single value and optimised using torch.optim.SGD() and torch.Tensor()
# This code is to be used as comparison with basicfitter.py

import torch
import torch.optim as optim
from tensorops.tensorutils import LossPlotter

if __name__ == "__main__":
    target = torch.tensor(10.0, dtype=torch.float64, requires_grad=False)
    x = torch.tensor(0.5, dtype=torch.float64, requires_grad=False)
    w_0 = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    w_1 = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    w_2 = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    def polynomial(w0, w1, w2, x):  # predictor function
        return w0 + w1 * x + w2 * (x**2)

    mse_loss = torch.nn.MSELoss()
    loss_plot = LossPlotter()
    optimizer = optim.SGD([w_0, w_1, w_2], lr=0.1)

    for gen in range(10):
        optimizer.zero_grad()
        polynomial_result = polynomial(w_0, w_1, w_2, x)
        loss = mse_loss(polynomial_result, target)
        loss.backward()

        print(f"generation {gen}: {loss.item():2f}")

        optimizer.step()
        loss_plot.register_datapoint(loss.item(), "ax^2+bx+c (PyTorch)")
    loss_plot.plot()
