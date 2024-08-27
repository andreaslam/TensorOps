from tensorops.tensorutils import LossPlotter
from tensorops.loss import MSELoss
from tensorops.model import Model
from tensorops.node import Node, backward, forward
from tensorops.optim import SGD, Adam


class LinearModel(Model):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        with self.context:
            self.m = Node(0.7, requires_grad=True, weight=True)
            self.c = Node(-0.3, requires_grad=True, weight=True)
            self.output_node = self.m * self.inputs + self.c
            self.loss = loss_criterion.loss(self.targets, self.output_node)

    def forward(self, input_node):
        with self.context:
            self.inputs.set_value(input_node.value)
            forward(self.context.nodes)
            return self.output_node

    def calculate_loss(self, output, target):
        with self.context:
            self.output_node.set_value(output.value)
            self.targets.set_value(target.value)
            return self.loss


class QuadraticModel(Model):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        with self.context:
            self.a = Node(0.7, requires_grad=True, weight=True)
            self.b = Node(-0.3, requires_grad=True, weight=True)
            self.c = Node(0.3, requires_grad=True, weight=True)
            self.output_node = (
                self.a * (self.inputs**2) + self.b * self.inputs + self.c
            )
            self.loss = loss_criterion.loss(self.targets, self.output_node)

    def forward(self, input_node):
        with self.context:
            self.inputs.set_value(input_node.value)
            forward(self.context.nodes)
            return self.output_node

    def calculate_loss(self, output, target):
        with self.context:
            self.output_node.set_value(output.value)
            self.targets.set_value(target.value)
            return self.loss


class CubicModel(Model):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        with self.context:
            self.a = Node(0.7, requires_grad=True, weight=True)
            self.b = Node(-0.3, requires_grad=True, weight=True)
            self.c = Node(0.3, requires_grad=True, weight=True)
            self.d = Node(-0.7, requires_grad=True, weight=True)
            self.output_node = (
                self.a * (self.inputs**3)
                + self.b * (self.inputs**2)
                + self.c * self.inputs
                + self.d
            )
            self.loss = loss_criterion.loss(self.targets, self.output_node)

    def forward(self, input_node):
        with self.context:
            self.inputs.set_value(input_node.value)
            forward(self.context.nodes)
            return self.output_node

    def calculate_loss(self, output, target):
        with self.context:
            self.output_node.set_value(output.value)
            self.targets.set_value(target.value)
            return self.loss


def train_model(model, optim, num_iterations, loss_plot):
    for i in range(num_iterations):
        model.zero_grad()
        output = model(Node(2.0, requires_grad=False))
        backward(model.context.nodes)
        loss = model.calculate_loss(output, Node(1.0, requires_grad=False))
        optim.step()
        print(f"generation {i}: loss: {loss.value}")
        loss_plot.register_datapoint(loss.value, f"{type(model).__name__}-TensorOps")

    loss_plot.plot()
    print(model)


if __name__ == "__main__":
    criterion = MSELoss()
    models = [LinearModel(criterion), QuadraticModel(criterion), CubicModel(criterion)]
    loss_plot = LossPlotter()
    for model in models:
        optim = Adam(model.get_weights(), lr=5e-3)
        train_model(model, optim, 100, loss_plot)
