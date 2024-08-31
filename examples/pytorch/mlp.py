from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorops.tensorutils import PlotterUtil


class MLP(nn.Module):
    def __init__(self, num_layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Linear(2, 2)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = self.sigmoid(layer(x))
        return x


def prepare_training_inputs():
    X, y = make_moons(n_samples=1000, noise=0.4, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    return X_train,X_test,y_train,y_test

def run_training_loop(X_train, y_train, model, loss_criterion, optimiser, loss_plot, num_epochs):
    for _ in tqdm(range(num_epochs)):
        model.train()
        optimiser.zero_grad()
        outputs = model(X_train)
        loss = loss_criterion(outputs, y_train)
        loss_plot.register_datapoint(loss.item(), f"{type(model).__name__}-PyTorch")
        loss.backward()
        optimiser.step()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_training_inputs()

    model = MLP(num_layers=3)

    loss_criterion = nn.MSELoss()

    optimiser = optim.Adam(model.parameters(), lr=1e-1)
    loss_plot = PlotterUtil()
    num_epochs = 1000

    run_training_loop(X_train, y_train, model, loss_criterion, optimiser, loss_plot, num_epochs)

    model.eval()
    
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = loss_criterion(y_pred, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")

    loss_plot.plot()
