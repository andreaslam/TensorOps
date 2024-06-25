import functools
from node import Node, NodeContext
import pickle


class Model:
    def __init__(self, weights=None, *args, **kwargs):
        if weights is None:
            weights = []
        self.weights = weights  # should be of type List[Node()]
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        pass

    # remove x from the execution nodes

    def __repr__(self):
        if self.weights:
            return f"Model(weights={self.weights})"
        return "[Warning]: no weights initialised yet"

    def load(self, path):
        with open(path, "rb") as f:
            self.weights = pickle.load(f)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.weights, f)


def softmax(x, dim=0):
    numerator = [num.exp() for num in x]
    denominator = sum(numerator)
    return numerator / denominator


def sigmoid(x):
    return Node(1) / (Node(1) + Node(1)/(x.exp()))
