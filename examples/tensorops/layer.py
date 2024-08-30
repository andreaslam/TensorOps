# Build a neural network layer using `tensorops.model.Layer`


from tensorops.model import Layer
from tensorops.node import NodeContext, Node, sigmoid
import random


def init_network_params(num_input_nodes, num_output_nodes):
    weights = [
        [
            Node(random.uniform(-1, 1), requires_grad=True, weight=True)
            for _ in range(num_input_nodes)
        ]
        for _ in range(num_output_nodes)
    ]
    bias = [
        Node(random.uniform(-1, 1), requires_grad=True, weight=True)
        for _ in range(num_output_nodes)
    ]

    return weights, bias


if __name__ == "__main__":
    random.seed(42)

    num_input_nodes = 3
    num_output_nodes = 3

    weights, bias = init_network_params(num_input_nodes, num_output_nodes)

    X = [
        Node(random.uniform(-1, 1), requires_grad=False, weight=False)
        for _ in range(num_input_nodes)
    ]

    layer = Layer(
        NodeContext(), num_input_nodes, num_output_nodes, sigmoid, weights, bias
    )

    print("Layer weights:", layer.weights)
    print("Layer biases:", layer.bias)

    y = layer(X)

    print(f"outputs: {[output.activation_output.value for output in y]}")
