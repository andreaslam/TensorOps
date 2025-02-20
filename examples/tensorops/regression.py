# Create and train a fully connected multi-layer perceptron (MLP) using `tensorops.Model`


from tqdm import tqdm
from tensorops.utils.data import prepare_dataset
from tensorops.loss import MSELoss
from tensorops.utils.models import SequentialModel
from tensorops.node import relu, sigmoid
from tensorops.optim import AdamW
from tensorops.utils.tensorutils import PlotterUtil
import random


class MLP(SequentialModel):
    def __init__(
        self,
        loss_criterion,
        num_input_nodes,
        num_output_nodes,
        num_hidden_layers,
        num_hidden_nodes,
        activation_function=sigmoid,
    ):
        super().__init__(loss_criterion)
        self.activation_function = activation_function
        self.num_hidden_layers = num_hidden_layers
        with self.context:
            self.add_layer(num_input_nodes, num_hidden_nodes, self.activation_function)
            for _ in range(self.num_hidden_layers):
                self.add_layer(
                    num_hidden_nodes, num_hidden_nodes, self.activation_function
                )
            self.add_layer(num_hidden_nodes, num_output_nodes, self.activation_function)
            self.loss = self.loss_criterion(
                self.targets, self.model_output_layer.layer_output_nodes
            )


def training_loop(X_train, y_train, mlp, optim, loss_plot, num_epochs):
    for _ in range(num_epochs):
        for X, y in tqdm(zip(X_train, y_train), desc="Training MLP"):
            mlp.zero_grad()
            outputs = mlp(X)
            loss = mlp.calculate_loss(outputs, y)
            mlp.backward()
            loss_plot.register_datapoint(loss.value, f"{type(mlp).__name__}-TensorOps")
            optim.step()


if __name__ == "__main__":
    random.seed(42)

    X, y = prepare_dataset("tensorops")

    num_epochs = 1
    num_input_nodes = len(X[0])
    num_hidden_nodes = 64
    num_hidden_layers = 8
    num_output_nodes = len(y[0])

    num_datapoints = len(X)

    model = MLP(
        MSELoss(),
        num_input_nodes,
        num_output_nodes,
        num_hidden_layers,
        num_hidden_nodes,
        relu,
    )

    optim = AdamW(model.get_weights(), lr=1e-3)

    loss_plot = PlotterUtil()

    training_loop(
        X,
        y,
        model,
        optim,
        loss_plot,
        num_epochs,
    )

    loss_plot.plot()
