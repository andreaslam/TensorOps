# Create and train a fully connected multi-layer perceptron (MLP) using `tensorops.Model`


from tqdm import tqdm
from tensorops.loss import MSELoss
from tensorops.model import Model
from tensorops.node import Node, sigmoid
from tensorops.optim import SGD
from tensorops.utils.tensorutils import PlotterUtil
import random


class MLP(Model):
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
                self.targets, self.output_layer.layer_output_nodes
            )


def training_loop(X, y, mlp, optim, loss_plot, num_epochs):
    for _ in tqdm(range(num_epochs), desc="Training network"):
        mlp.zero_grad()
        outputs = mlp(X)
        loss = mlp.calculate_loss(outputs, y)
        loss_plot.register_datapoint(loss.value, f"{type(mlp).__name__}-TensorOps")
        mlp.backward()
        optim.step()


if __name__ == "__main__":
    random.seed(42)

    num_epochs = 100
    num_input_nodes = 2
    num_hidden_nodes = 12
    num_hidden_layers = 3
    num_output_nodes = 2

    X = [
        Node(random.uniform(-2, 2), requires_grad=False) for _ in range(num_input_nodes)
    ]
    y = [
        Node(random.uniform(-2, 2), requires_grad=False)
        for _ in range(num_output_nodes)
    ]
    model = MLP(
        MSELoss(),
        num_input_nodes,
        num_output_nodes,
        num_hidden_layers,
        num_hidden_nodes,
        sigmoid,
    )
    optim = SGD(model.get_weights(), lr=1e-1)

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
