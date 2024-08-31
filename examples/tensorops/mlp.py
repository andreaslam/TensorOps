# creates a fully connected multi-layer perceptron (MLP) using `tensorops.Model`

from tqdm import tqdm
from tensorops.loss import MSELoss
from tensorops.model import Model
from tensorops.node import Node, backward, forward, sigmoid
from tensorops.optim import SGD, Adam
import random

from tensorops.tensorutils import PlotterUtil, visualise_graph


class MLP(Model):
    def __init__(self, loss_criterion, num_layers):
        super().__init__(loss_criterion)
        for _ in range(num_layers):
            self.add_layer(10, 10, sigmoid)
        self.output_nodes = [
            activation.activation_output
            for activation in self.model_layers[-1].layer_output
        ]
        self.input_nodes = [
            Node(0.0, requires_grad=False)
            for _ in range(self.model_layers[0].num_input_nodes)
        ]
        self.targets = [
            Node(0.0, requires_grad=False) for _ in range(len(self.output_nodes))
        ]
        self.loss = loss_criterion.loss(self.targets, self.output_nodes)

    def forward(self, model_inputs):
        input_node = model_inputs
        for layer in self.model_layers:
            assert len(input_node) == layer.num_input_nodes
            input_node = layer(input_node)
        return self.output_nodes

    def calculate_loss(self, output, target):
        with self.context:
            return self.loss_criterion.loss(output, target)


def generate_training_data():
    X_train = [
        [
            Node(random.uniform(-1, 1), requires_grad=False, weight=False)
            for _ in range(10)
        ]
        for _ in range(100)
    ]
    y_train = [
        [
            Node(random.uniform(-1, 1), requires_grad=False, weight=False)
            for _ in range(10)
        ]
        for _ in range(100)
    ]
    return X_train, y_train


def training_loop(X_train, y_train, linear_model, optim, loss_plot):
    for X, y in zip(X_train, y_train):
        linear_model.zero_grad()
        output = linear_model(X)
        loss = linear_model.calculate_loss(output, y)
        print(loss)
        backward(linear_model.context.nodes)
        optim.step()
        loss_plot.register_datapoint(
            loss.value, f"{type(linear_model).__name__}-TensorOps"
        )


def plot_model_output(X_train, linear_model, graph_plot):
    for x in X_train:
        graph_plot.register_datapoint(
            linear_model(x).value, x=x.value, label="y=mx+c (TensorOps)"
        )


if __name__ == "__main__":
    random.seed(42)
    model = MLP(MSELoss(), 10)
    num_epochs = 10
    X_train, y_train = generate_training_data()

    optim = SGD(model.get_weights(), lr=1e-2)

    loss_plot = PlotterUtil()

    for _ in tqdm(range(100), desc=f"Training {type(model).__name__}-TensorOps"):
        training_loop(X_train, y_train, model, optim, loss_plot)

    loss_plot.plot()
