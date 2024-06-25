import math


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
