import math
from abc import ABC


class Node(ABC):

    """
    `tensorops.Node` is a node in a computational graph, containing its value and gradient.

    Attributes
    ----------
    value (float): The value of the node to be created.
    parents (List[tensorops.Nodes]): A list of containing parent nodes.
    children (List[tensorops.Nodes]): A list of containing children nodes.
    requires_grad (bool): Whether the node requires gradient tracking.
    weight (bool): Whether the node is a neural network weight.
    current_context (Optional[tensorops.NodeContext]): Manages the operational context for nodes during computation.
    """

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

    def __abs__(self):
        return Abs(self)

    def __repr__(self):
        if not self.value:
            return f"{type(self).__name__}(value=None, grad={self.grad:.5f}, weight={self.weight})"
        return f"{type(self).__name__}(value={self.value:.5f}, grad={self.grad:.5f}, weight={self.weight})"

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

    def set_value(self, new_value):
        self.value = new_value
        self.trigger_recompute()

    def trigger_recompute(self):
        if NodeContext.current_context:
            NodeContext.current_context.recompute()


class NodeContext:

    """
    `tensorops.NodeContext` manages the operational context for nodes during computation.
    """

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
        """
        Creates a node to be added to the computational graph stored in `tensorops.NodeContext.context`
        Args:
            node (tensorops.Node): The node object to be added to the computational graph
        """

        self.nodes.append(node)
    def recompute(self):
        forward(self.nodes)

    def weights_enabled(self):
        return [node for node in self.nodes if node.weight]

    def grad_enabled(self):
        return [node for node in self.nodes if node.requires_grad]

    def __repr__(self) -> str:
        return str([node for node in self.nodes])


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


class Abs(Node):
    def __init__(self, node1):
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = abs(self.node1.value)

    def get_grad(self):
        if self.value != 0:
            self.node1.grad += self.value / abs(self.value)


class Sigmoid(Node):
    def __init__(self, node1):
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self):
        self.value = 1 / (1 + math.exp(-self.node1.value))

    def get_grad(self):
        sigmoid_value = self.value
        self.node1.grad += self.grad * sigmoid_value * (1 - sigmoid_value)


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
