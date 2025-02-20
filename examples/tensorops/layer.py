# Build and train a neural network layer using `tensorops.model.Layer`


from tensorops.loss import MSELoss
from tensorops.model import Layer
from tensorops.node import NodeContext, Node, backward, sigmoid, zero_grad
import random

from tensorops.optim import Adam
from tensorops.utils.tensorutils import PlotterUtil


def init_network_params(num_input_nodes, num_output_nodes):
    weights = [
        [random.uniform(-1, 1) for _ in range(num_input_nodes)]
        for _ in range(num_output_nodes)
    ]
    bias = [random.uniform(-1, 1) for _ in range(num_output_nodes)]

    return weights, bias


class LayerTest(Layer):
    def __init__(
        self,
        context,
        num_input_nodes,
        num_output_nodes,
        activation_function,
        seed=None,
        output_weights=None,
        output_bias=None,
        loss_criterion=MSELoss(),
    ):
        super().__init__(
            context,
            num_input_nodes,
            num_output_nodes,
            activation_function,
            seed,
            layer_input_nodes=None,
            output_weights=output_weights,
            output_bias=output_bias,
        )
        with self.context:
            self.loss_criterion = loss_criterion
            self.targets = [
                Node(0.0, requires_grad=False) for _ in range(self.num_output_nodes)
            ]
            self.loss = self.loss_criterion(self.targets, self.layer_output_nodes)

    def calculate_loss(self, output, target):
        with self.context:
            for layer_target_nodes, training_target, layer_output_nodes, y in zip(
                self.targets, target, self.layer_output_nodes, output
            ):
                layer_output_nodes.set_value(y.value)
                layer_target_nodes.set_value(training_target.value)
        return self.loss


if __name__ == "__main__":
    random.seed(42)

    num_input_nodes = 3
    num_output_nodes = 3
    num_epochs = 10

    num_data = 10

    X_train = [
        [
            Node(random.uniform(-0.1, 0.1), requires_grad=False)
            for _ in range(num_input_nodes)
        ]
        for _ in range(num_data)
    ]
    y_train = [
        [
            Node(random.uniform(-0.1, 0.1), requires_grad=False)
            for _ in range(num_output_nodes)
        ]
        for _ in range(num_data)
    ]
    weights, bias = init_network_params(num_input_nodes, num_output_nodes)

    loss_plot = PlotterUtil()

    layer = LayerTest(
        NodeContext(),
        num_input_nodes,
        num_output_nodes,
        sigmoid,
        output_weights=weights,
        output_bias=bias,
    )

    optim = Adam(layer.context.weights_enabled(), lr=7e-2)

    for _ in range(num_epochs):
        for X, y in zip(X_train, y_train):
            zero_grad(layer.context.nodes)
            y_preds = layer(X)
            loss = layer.calculate_loss(y_preds, y)
            backward(layer.context.nodes)
            optim.step()
            loss_plot.register_datapoint(
                loss.value, f"{type(layer).__name__}-TensorOps"
            )

    loss_plot.plot()
