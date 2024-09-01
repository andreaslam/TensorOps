# Given three polynomial fitters constructed using the tensorops.Model base class and fits to the point (2,1).
# The code will train each model for 100 "epochs" and return the best performing fitter with its respective loss.


from tensorops.utils.tensorutils import PlotterUtil
from tensorops.loss import MSELoss
from tensorops.model import Model
from tensorops.node import Node, backward, forward
from tensorops.optim import Adam
from tqdm import tqdm
from helpers import SimpleModel


class LinearModel(SimpleModel):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        with self.context:
            self.m = Node(0.7, requires_grad=True, weight=True)
            self.c = Node(0.3, requires_grad=True, weight=True)
            self.output_node = self.m * self.input_node + self.c
            self.loss = loss_criterion.loss(self.targets, self.output_node)


class QuadraticModel(SimpleModel):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        with self.context:
            self.a = Node(0.7, requires_grad=True, weight=True)
            self.b = Node(0.3, requires_grad=True, weight=True)
            self.c = Node(0.3, requires_grad=True, weight=True)
            self.output_node = (
                self.a * (self.input_node**2) + self.b * self.input_node + self.c
            )
            self.loss = loss_criterion.loss(self.targets, self.output_node)


class CubicModel(SimpleModel):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        with self.context:
            self.a = Node(0.7, requires_grad=True, weight=True)
            self.b = Node(0.3, requires_grad=True, weight=True)
            self.c = Node(0.3, requires_grad=True, weight=True)
            self.d = Node(-0.7, requires_grad=True, weight=True)
            self.output_node = (
                self.a * (self.input_node**3)
                + self.b * (self.input_node**2)
                + self.c * self.input_node
                + self.d
            )
            self.loss = loss_criterion.loss(self.targets, self.output_node)


def train_model(model, optim, num_iterations, loss_plot, results):
    for _ in tqdm(
        range(num_iterations), desc=f"Training {type(model).__name__}-TensorOps"
    ):
        model.zero_grad()
        output = model(Node(2.0, requires_grad=False))
        loss = model.calculate_loss(output, Node(1.0, requires_grad=False))
        model.backward()
        optim.step()
        loss_plot.register_datapoint(loss.value, f"{type(model).__name__}-TensorOps")
    results[f"{type(model).__name__}-TensorOps"] = loss.value
    return results


if __name__ == "__main__":
    criterion = MSELoss()
    models = [LinearModel(criterion), QuadraticModel(criterion), CubicModel(criterion)]
    loss_plot = PlotterUtil()
    results = {}
    for model in models:
        optim = Adam(model.get_weights(), lr=5e-3)
        train_model(model, optim, 100, loss_plot, results)
    print("Training results:", results)
    print(
        f"Best model was {min(results, key=results.get)} with loss {results[min(results, key=results.get)]}"
    )
    loss_plot.plot()
