import torch
import random
from main import LossPlotter


# Define the polynomial function
def polynomial(x, a, b, c):
    return a * x**2 + b * x + c


# Function to add noise to data
def add_noise(data):
    noise = torch.tensor(random.uniform(-100, 100), dtype=torch.float64)
    return data + noise


if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)

    # Loss plotter
    loss_plot = LossPlotter()

    # Training data
    x_values = torch.tensor(
        [random.uniform(-10, 10) for _ in range(10)], dtype=torch.float64
    )
    y_train = torch.tensor(
        [add_noise(polynomial(x, 2, 20, -1)) for x in x_values], dtype=torch.float64
    )  # Noisy version of polynomial
    print(x_values, y_train)

    # Model parameters (weights)
    a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)

    # Loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam([a, b, c], lr=1e-5)

    # Training loop
    for epoch in range(100):
        for x, y in zip(x_values, y_train):
            optim.zero_grad()

            y_pred = polynomial(x, a, b, c)
            print(x.item(), y_pred.item())
            loss = loss_fn(y_pred.unsqueeze(0), y.unsqueeze(0))
            print(loss.item())

            # Backward pass
            loss.backward()

            # Optimize
            optim.step()

            # Logging
            loss_plot.register_datapoint(loss.item(), label="loss-pytorch")

    # Plot the loss
    loss_plot.plot()
