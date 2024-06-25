from node import Node
import pickle


class Model:
    def __init__(self, weights, *args, **kwargs):
        if weights is None:
            # TODO:
            self.weights = self.random_initialise()
        else:
            self.weights = weights
        self.args = args
        self.kwargs = kwargs

    def random_initialise(self):
        # TODO impl random initialise
        return [0, 0, 0]  # dummy output

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def __repr__(self):
        if self.weights:
            return f"Model containing weights: {self.weights}"
        return "[Warning]: No weights initialised yet."

    def load(self, path: str):
        with open(path, "rb") as f:
            self.weights = pickle.load(f)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.weights, f)


def softmax(x, dim=0):
    numerator = [num.exp() for num in x]
    denominator = sum(numerator)
    return numerator / denominator


def sigmoid(x):
    return Node(1) / (Node(1) + x.exp())
