# Build an artificial neuron using `tensorops.model.Activation`


from tensorops.model import Activation
from tensorops.node import NodeContext, Node, sigmoid
import random


if __name__ == "__main__":
    random.seed(42)
    num_inputs = 10
    weights = [
        Node(random.uniform(-1, 1), requires_grad=True, weight=True) for _ in range(10)
    ]

    bias = Node(random.uniform(-1, 1), requires_grad=True, weight=True)

    activation = Activation(num_inputs, sigmoid, NodeContext(), weights, bias)
    print(f"Weights: {activation.weights},\nBias: {activation.bias}")
    X = [
        Node(random.uniform(-10, 10), requires_grad=True, weight=True)
        for _ in range(10)
    ]

    y = activation(X)
    print("Output:", y.value)
