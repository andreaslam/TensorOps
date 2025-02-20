# Build and train an artificial neuron using `tensorops.model.Activation`


from tensorops.loss import MSELoss
from tensorops.model import Activation
from tensorops.node import NodeContext, Node, backward, sigmoid, zero_grad
from tensorops.optim import SGD
from tensorops.utils.tensorutils import PlotterUtil
import random


class TestActivation(Activation):
    def __init__(
        self,
        num_input_nodes,
        activation_function,
        context,
        weights=None,
        bias=None,
        seed=None,
        input_nodes=None,
        loss_criterion=MSELoss(),
    ):
        super().__init__(
            context,
            num_input_nodes,
            activation_function,
            weights,
            bias,
            seed,
            input_nodes,
        )
        with self.context:
            self.loss_criterion = loss_criterion
            self.target = Node(0.0, requires_grad=False)
            self.loss = self.loss_criterion(self.target, self.activation_output)

    def calculate_loss(self, output, target):
        assert self.activation_output, "Output behaviour not defined"
        with self.context:
            self.activation_output.set_value(output.value)
            self.target.set_value(target.value)
            return self.loss


if __name__ == "__main__":
    random.seed(42)

    num_inputs = 10
    num_epochs = 10

    num_data = 10

    weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
    bias = random.uniform(-1, 1)
    print(f"Weights: {weights},\nBias: {bias}")

    X_train = [
        [
            Node(random.uniform(-0.1, 0.1), requires_grad=False)
            for _ in range(num_inputs)
        ]
        for _ in range(num_data)
    ]
    y_train = [Node(0.0, requires_grad=False) for _ in range(num_data)]

    activation = TestActivation(num_inputs, sigmoid, NodeContext(), weights, bias)

    optim = SGD(activation.context.weights_enabled(), lr=1e-1)

    loss_plot = PlotterUtil()

    for _ in range(num_epochs):
        for X, y in zip(X_train, y_train):
            zero_grad(activation.context.nodes)
            y_preds = activation(X)
            loss = activation.calculate_loss(y_preds, y)
            backward(activation.context.nodes)
            optim.step()
            loss_plot.register_datapoint(
                loss.value, f"{type(activation).__name__}-TensorOps"
            )

    loss_plot.plot(display=False)
