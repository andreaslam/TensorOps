# creates a fully connected multi-layer perceptron (MLP) using `tensorops.Model`

from tqdm import tqdm
from tensorops.loss import MSELoss
from tensorops.model import Model
from tensorops.node import Node, sigmoid
from tensorops.optim import Adam
import random


class MLP(Model):
    def __init__(self, loss_criterion, num_layers):
        super().__init__(loss_criterion)
        for _ in range(num_layers):
            self.add_layer(10, 10, sigmoid)
        self.output_nodes = [
            activation.activation_output
            for activation in self.model_layers[-1].layer_output
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
        self.targets = [node.set_value(o) for node, o in zip(self.targets, output)]
        self.output_nodes = [
            node.set_value(t) for node, t in zip(self.output_nodes, target)
        ]
        return self.loss


if __name__ == "__main__":
    random.seed(42)
    model = MLP(MSELoss(), 10)
    X = [
        Node(random.uniform(-1, 1), requires_grad=False, weight=False)
        for _ in range(10)
    ]
    y = model(X)
    print(y)
