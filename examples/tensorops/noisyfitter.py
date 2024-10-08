# Given an equation of a line (y = mx + c) and random inputs from (-5,5) the linear neural network, built and trained using PyTorch, will try and fit to the training data.
# There is random noise added to the resulting y value of the equation. This is to test the model's ability to adjust its weights for a line of best fit.


import random
from tqdm import tqdm
from tensorops.node import Node
from tensorops.loss import MSELoss
from tensorops.optim import SGD
from tensorops.utils.tensorutils import PlotterUtil, visualise_graph
from tensorops.utils.models import SimpleSequentialModel


class LinearModel(SimpleSequentialModel):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        with self.context:
            self.m = Node(0.6, requires_grad=True, weight=True)
            self.c = Node(0.7, requires_grad=True, weight=True)
            self.output_node = self.m * self.input_node + self.c
            self.loss = self.loss_criterion.loss(self.targets, self.output_node)


def plot_training_data(X_train, y_train, graph_plot):
    for x, y in zip(X_train, y_train):
        graph_plot.register_datapoint(
            datapoint=y.value,
            label="Training Data (TensorOps)",
            x=x.value,
            plot_style="scatter",
            colour="red",
        )


def training_loop(X_train, y_train, linear_model, optim, loss_plot):
    for X, y in zip(X_train, y_train):
        linear_model.zero_grad()
        output = linear_model(X)
        loss = linear_model.calculate_loss(output, y)
        linear_model.backward()
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

    target_m = 2.0
    target_c = 10.0

    X_train = [
        Node(random.uniform(-5, 5), requires_grad=False, weight=False)
        for _ in range(10)
    ]

    y_train = [
        Node(
            x.value * target_m + target_c + random.uniform(-1, 1),
            requires_grad=False,
            weight=False,
        )
        for x in X_train
    ]

    linear_model = LinearModel(
        MSELoss(),
    )

    visualise_graph(linear_model.context.nodes)

    optim = SGD(linear_model.get_weights(), lr=1e-2)

    loss_plot = PlotterUtil()

    graph_plot = PlotterUtil(x_label="x", y_label="y")

    plot_training_data(X_train, y_train, graph_plot)

    for _ in tqdm(range(100), desc=f"Training {type(linear_model).__name__}-TensorOps"):
        training_loop(X_train, y_train, linear_model, optim, loss_plot)

    print(linear_model)
    loss_plot.plot()

    plot_model_output(X_train, linear_model, graph_plot)

    graph_plot.plot()
