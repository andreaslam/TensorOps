import math
import matplotlib.pyplot as plt
import networkx as nx
import random
import pickle


class Node:
    def __init__(self, value, requires_grad=True, weight=False):
        self.value = value
        self.grad = 0
        self.parents = []
        self.children = []
        self.requires_grad = requires_grad
        self.weight = weight
        if NodeContext.current_context is not None:
            NodeContext.current_context.add_node(self)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __pow__(self, index):
        return Power(self, index)

    def __repr__(self):
        return f"Node(value={self.value}, grad={self.grad})"

    def sin(self):
        return Sin(self)

    def cos(self):
        return Cos(self)

    def tanh(self):
        return Tanh(self)

    def ReLU(self):
        return ReLU(self)

    def negate(self):
        return Negate(self)

    def exp(self):
        return Exp(self)

    def compute(self):
        pass

    def get_grad(self):
        pass

    def zero_grad(self):
        self.grad = 0

    def seed_grad(self, seed):
        self.grad = seed


class LossPlotter:
    def __init__(self):
        self.datapoints = {}
        self.labels = []
        self.xs = {}

    def register_datapoint(self, datapoint, label, x=None):
        if label not in self.labels:
            self.labels.append(label)
            self.datapoints[label] = []
            self.xs[label] = []

        self.datapoints[label].append(datapoint)

        if x is not None:
            self.xs[label].append(x)
        else:
            self.xs[label].append(len(self.datapoints[label]) - 1)

    def plot(self):
        for label in self.labels:
            plt.plot(self.xs[label], self.datapoints[label], label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.close()


class NodeContext:
    current_context = None

    def __init__(self):
        self.nodes = []

    def __enter__(self):
        self.prev_context = NodeContext.current_context
        NodeContext.current_context = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        NodeContext.current_context = self.prev_context

    def add_node(self, node):
        self.nodes.append(node)

    def weights_enabled(self):
        return [node for node in self.nodes if node.weight]

    def grad_enabled(self):
        return [node for node in self.nodes if node.requires_grad]


class Add(Node):
    def __init__(self, node1, node2):
        super().__init__(0)
        self.node1 = node1
        self.node2 = node2
        self.parents = [self.node1, self.node2]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = self.node1.value + self.node2.value

    def get_grad(self):
        self.node1.grad += self.grad * 1
        self.node2.grad += self.grad * 1


class Sub(Node):
    def __init__(self, node1, node2):
        super().__init__(0)
        self.node1 = node1
        self.node2 = node2
        self.parents = [self.node1, self.node2]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = self.node1.value - self.node2.value

    def get_grad(self):
        self.node1.grad += self.grad * 1
        self.node2.grad += self.grad * -1


class Mul(Node):
    def __init__(self, node1, node2):
        super().__init__(0)
        self.node1 = node1
        self.node2 = node2
        self.parents = [self.node1, self.node2]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = self.node1.value * self.node2.value

    def get_grad(self):
        self.node1.grad += self.grad * self.node2.value
        self.node2.grad += self.grad * self.node1.value


class Div(Node):
    def __init__(self, node1, node2):
        super().__init__(0)
        self.node1 = node1
        self.node2 = node2
        self.parents = [self.node1, self.node2]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = self.node1.value / self.node2.value

    def get_grad(self):
        self.node1.grad += self.grad * 1 / self.node2.value
        self.node2.grad += self.grad * -self.node1.value / (self.node2.value**2)


class Power(Node):
    def __init__(self, node1, idx):
        super().__init__(0)
        self.node1 = node1
        self.idx = idx
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = self.node1.value**self.idx

    def get_grad(self):
        self.node1.grad += self.grad * self.idx * (self.node1.value ** (self.idx - 1))


class Sin(Node):
    def __init__(self, node1):
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = math.sin(self.node1.value)

    def get_grad(self):
        self.node1.grad += self.grad * math.cos(self.node1.value)


class Cos(Node):
    def __init__(self, node1):
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = math.cos(self.node1.value)

    def get_grad(self):
        self.node1.grad += self.grad * -math.sin(self.node1.value)


class Tanh(Node):
    def __init__(self, node1):
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = math.tanh(self.node1.value)

    def get_grad(self):
        self.node1.grad += self.grad * (1 - (math.tanh(self.node1.value) ** 2))


class ReLU(Node):
    def __init__(self, node1):
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = max(0, self.node1.value)

    def get_grad(self):
        if self.node1.value > 0:
            self.node1.grad += self.grad * 1
        elif self.node1.value < 0:
            self.node1.grad += self.grad * 0


class Negate(Node):
    def __init__(self, node1):
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = self.node1.value * -1

    def get_grad(self):
        self.node1.grad += self.grad * -1


class Exp(Node):
    def __init__(self, node1):
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = math.exp(self.node1.value)

    def get_grad(self):
        self.node1.grad += self.grad * self.value


class Optim:
    def __init__(self, lr=1e-3, maximise=False, weight_decay=0.0):
        self.lr = lr
        self.maximise = maximise
        self.weight_decay = weight_decay

    def step(self):
        pass


class Adam(Optim):
    def __init__(
        self,
        parameters,
        lr=1e-3,
        maximise=False,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    ):
        super().__init__(lr, maximise, weight_decay)
        self.parameters = parameters
        self.t = 0
        self.betas = betas
        self.m = {param: 0 for param in parameters}
        self.v = {param: 0 for param in parameters}
        self.eps = eps
        self.amsgrad = amsgrad
        self.v_hat_max = {param: 0 for param in parameters}

    def step(self):
        self.t += 1
        for param in filter(lambda p: p.requires_grad, self.parameters):
            g_t = -param.grad if self.maximise else param.grad

            if self.weight_decay != 0.0:
                g_t += self.weight_decay * param.value

            self.m[param] = self.betas[0] * self.m[param] + (1 - self.betas[0]) * g_t
            self.v[param] = self.betas[1] * self.v[param] + (1 - self.betas[1]) * (
                g_t**2
            )

            m_hat_t = self.m[param] / (1 - self.betas[0] ** self.t)
            v_hat_t = self.v[param] / (1 - self.betas[1] ** self.t)

            if self.amsgrad:
                self.v_hat_max[param] = max(self.v_hat_max[param], v_hat_t)
                param.value -= (
                    self.lr * m_hat_t / (math.sqrt(self.v_hat_max[param]) + self.eps)
                )
            else:
                param.value -= self.lr * m_hat_t / (math.sqrt(v_hat_t) + self.eps)


class SGD(Optim):
    def __init__(
        self,
        parameters,
        lr=1e-3,
        maximise=False,
        weight_decay=0.0,
        nesterov=False,
        dampening=0,
        momentum=0,
    ):
        super().__init__(lr, maximise, weight_decay)
        self.parameters = parameters
        self.t = 0
        self.dampening = dampening
        self.nesterov = nesterov
        self.momentum = momentum
        self.b_t = {param: 0 for param in parameters}

    def step(self):
        self.t += 1
        for param in filter(lambda p: p.requires_grad, self.parameters):
            param.value -= self.lr * param.grad

            g_t = param.grad
            if self.weight_decay != 0.0:
                g_t = g_t + self.weight_decay * param.value

            if self.momentum != 0:
                if self.t > 1:
                    self.b_t[param] = (
                        self.momentum * self.b_t[param] + (1 - self.dampening) * g_t
                    )
                else:
                    self.b_t[param] = g_t

                if self.nesterov:
                    g_t = g_t + self.momentum * self.b_t[param]
                else:
                    g_t = self.b_t[param]

            if self.maximise:
                param.value = param.value + self.lr * g_t
            else:
                param.value = param.value - self.lr * g_t


class Loss:
    def __init__(self):
        pass

    def loss(self):
        pass

    def backward(self, context):
        context.nodes[-1].seed_grad(
            1
        )  # seed the gradient of the last node (output node) as 1
        for node in reversed(context.nodes):
            if node.requires_grad:
                node.get_grad()


class L1Loss(Loss):
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):
        return abs(actual - target)


class MSELoss(Loss):  # L2 loss
    def __init__(self):
        super().__init__()

    def loss(self, actual, target):
        return (target - actual) ** 2


def visualize_graph(nodes):
    G = nx.DiGraph()
    labels = {}
    for node in nodes:
        node_id = id(node)
        node_label = f"{type(node).__name__}\nVal: {round(node.value,2)}\nGrad: {round(node.grad,2)}"
        labels[node_id] = node_label
        G.add_node(node_id)
        for parent in node.parents:
            parent_id = id(parent)
            G.add_edge(parent_id, node_id)

    pos = nx.planar_layout(G)
    colourmap = [
        "#00B4D9" if node.requires_grad else "#C1E1C1" for (_, node) in zip(G, nodes)
    ]
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=800,
        node_color=colourmap,
        font_size=6,
    )
    plt.show()


def forward(nodes):
    for node in nodes:
        node.compute()


def backward(nodes):
    nodes[-1].seed_grad(1)
    for node in reversed(nodes):
        if node.requires_grad:
            node.get_grad()


def zero_grad(nodes):
    for node in nodes:
        node.zero_grad()


def polynomial(x, a, b, c):
    return a * x**2 + b * x + c


def add_noise(data):
    noise = random.uniform(-100, 100)
    return data + noise



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
        pass

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

class PolynomialFitter(Model):
    def __init__(self, weights, *args, **kwargs):
        super().__init__(weights, *args, **kwargs)
        if len(self.weights) < 3:
            raise ValueError("PolynomialFitter requires at least 3 weights.")
        self.a = self.weights[0]
        self.b = self.weights[1]
        self.c = self.weights[2]

    def forward(self, x):
        return sigmoid(self.a * x**2 + self.b * x + self.c)

if __name__ == "__main__":
    with NodeContext() as context:
        m = PolynomialFitter(
            [
                Node(1.0, requires_grad=True, weight=True),
                Node(2.0, requires_grad=True, weight=True),
                Node(3.0, requires_grad=True, weight=True),
            ]
        )
        y_1 = m.forward(Node(3.0, requires_grad=False))
        print(y_1)
        visualize_graph(context.nodes)
        x_2 = [
            Node(1.0, requires_grad=False),
            Node(2.0, requires_grad=False),
            Node(3.0, requires_grad=False),
            Node(4.0, requires_grad=False),
            Node(5.0, requires_grad=False),
        ]
        y_2 = m.softmax(x_2)
        print(y_2)
