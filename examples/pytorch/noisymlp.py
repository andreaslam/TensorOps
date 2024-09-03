import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tensorops.utils.tensorutils import PlotterUtil
from helpers import init_network_params


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer
        self.fc3 = nn.Linear(64, 10)  # Output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = torch.relu(self.fc2(x))  # ReLU activation
        x = self.fc3(x)  # Output layer (no softmax, handled by loss function)
        return x


def training_loop(X_train, y_train, mlp, optimiser, loss_plot, num_epochs):
    criterion = nn.MSELoss()
    model.train()
    i = 0
    for _ in tqdm(range(num_epochs), desc="Training MNIST network"):
        for X, y in zip(X_train[:10], y_train[:10]):
            optimiser.zero_grad()
            outputs = model(X).to(torch.float32)
            loss = criterion(outputs, y).to(torch.float32)
            print(outputs, y, loss.item())
            loss.backward()
            loss_plot.register_datapoint(loss.item(), f"{type(mlp).__name__}-TensorOps")
            if i % 32 == 0:
                optimiser.step()
            i += 1


def load_data(file_paths):
    all_data = []
    for file_path in file_paths:
        all_data.append(torch.load(file_path, weights_only=False).to(torch.float32))
    return all_data


if __name__ == "__main__":
    random.seed(42)

    num_epochs = 100
    num_input_nodes = 784
    num_hidden_nodes = 64
    num_hidden_layers = 10
    num_output_nodes = 10
    folder_path = "/Users/andreas/Desktop/Code/RemoteFolder/TensorOps/examples/data/MNIST/processed_pytorch"
    files = [
        f"{folder_path}/test_data.pt",
        f"{folder_path}/test_labels.pt",
        f"{folder_path}/train_data.pt",
        f"{folder_path}/train_labels.pt",
    ]

    X_test, y_test, X_train, y_train = load_data(files)
    y_train = torch.tensor(
        [[1.0 if y.item() == i else 0.0 for i in range(10)] for y in y_train]
    )
    model = MNISTModel()
    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    loss_plot = PlotterUtil()

    training_loop(
        X_train,
        y_train,
        model,
        optimiser,
        loss_plot,
        num_epochs,
    )

    loss_plot.plot()
