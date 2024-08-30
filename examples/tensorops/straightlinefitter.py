# Given a model y = mx + c, fitters constructed using the tensorops.Model base class and fits to the equation y = 1
# The code will train each model for 100 "epochs" and return the best performing fitter with its respective loss.


import random
from tensorops.tensorutils import PlotterUtil
from tensorops.loss import MSELoss
from tensorops.model import Model
from tensorops.node import Node, backward, forward
from tensorops.optim import Adam
from tqdm import tqdm


class LinearModel(Model):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        with self.context:
            self.m = Node(0.7, requires_grad=True, weight=True)
            self.c = Node(0.3, requires_grad=True, weight=True)
            self.output_node = self.m * self.input_nodes + self.c
            self.loss = loss_criterion.loss(self.targets, self.output_node)

    def forward(self, input_node):
        with self.context:
            self.input_nodes.set_value(input_node.value)
            forward(self.context.nodes)
            return self.output_node

    def calculate_loss(self, output, target):
        with self.context:
            self.output_node.set_value(output.value)
            self.targets.set_value(target.value)
            return self.loss


def train_model(model, optim, num_iterations, loss_plot):
    random.seed(42)
    for _ in tqdm(
        range(num_iterations), desc=f"Training {type(model).__name__}-TensorOps"
    ):
        model.zero_grad()
        output = model(Node(random.randint(-5, 5), requires_grad=False, weight=False))
        loss = model.calculate_loss(
            output, Node(1.0, requires_grad=False, weight=False)
        )
        backward(model.context.nodes)
        optim.step()
        loss_plot.register_datapoint(loss.value, f"{type(model).__name__}-TensorOps")
    loss_plot.plot()
    print(model)


if __name__ == "__main__":
    criterion = MSELoss()
    model = LinearModel(criterion)
    loss_plot = PlotterUtil()
    optim = Adam(model.get_weights(), lr=5e-2)
    train_model(model, optim, 100, loss_plot)
