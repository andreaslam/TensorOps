from __future__ import annotations

import math
import pickle
import itertools
from typing import Any, Generator


class Node:

    """
    `tensorops.Node` is a node in a computational graph, containing its value and gradient.

    Attributes
    ----------
    value (float): The value of the node to be created.
    parents (list[tensorops.Nodes]): A list containing parent nodes.
    children (list[tensorops.Nodes]): A list containing children nodes.
    requires_grad (bool): Whether the node requires gradient tracking.
    weight (bool): Whether the node is a neural network weight.
    current_context (Optional[tensorops.NodeContext]): Manages the operational context for nodes during computation.
    """

    def __init__(
        self, value: float, requires_grad: bool = True, weight: bool = False
    ) -> None:
        value = float(value)
        assert isinstance(value, float), f"Datatype must be a float, not {type(value)}"

        self.value = value
        self.grad = 0.0
        self.parents = []
        self.children = []
        self.requires_grad = requires_grad
        self.weight = weight
        if NodeContext.current_context is not None:
            NodeContext.current_context.add_node(self)

    def __add__(self, other: int | float | Node) -> Add:
        return Add(self, other if isinstance(other, Node) else Node(other))

    def __mul__(self, other: int | float | Node) -> Mul:
        return Mul(self, other if isinstance(other, Node) else Node(other))

    def __sub__(self, other: int | float | Node) -> Sub:
        return Sub(self, other if isinstance(other, Node) else Node(other))

    def __truediv__(self, other: int | float | Node) -> Div:
        return Div(self, other if isinstance(other, Node) else Node(other))

    def __pow__(self, index: int | float | Node) -> Power:
        return Power(self, index if isinstance(index, Node) else Node(index))

    def __radd__(self, other):
        return Add(other if isinstance(other, Node) else Node(other), self)

    def __rsub__(self, other):
        return Sub(other if isinstance(other, Node) else Node(other), self)

    def __rmul__(self, other):
        return Mul(other if isinstance(other, Node) else Node(other), self)

    def __rtruediv__(self, other):
        return Div(other if isinstance(other, Node) else Node(other), self)

    def __floordiv__(self, other):
        return IntDiv(self, other if isinstance(other, Node) else Node(other))

    def __mod__(self, other):
        return Mod(self, other if isinstance(other, Node) else Node(other))

    def __abs__(self) -> Abs:
        return Abs(self)

    def __neg__(self) -> Mul:
        return Mul(self, Node(-1))

    def __repr__(self) -> str:
        if self.value == None:
            return f"{type(self).__name__}(value=None, grad={round(self.grad, 4)}, weight={self.weight})"
        return f"{type(self).__name__}(value={round(self.value, 4)}, grad={round(self.grad, 4)}, weight={self.weight})"

    def __float__(self) -> float:
        return 0.0 if not self.value else self.value

    def sin(self) -> Sin:
        return Sin(self)

    def cos(self) -> Cos:
        return Cos(self)

    def tanh(self) -> Tanh:
        return Tanh(self)

    def relu(self) -> ReLU:
        return ReLU(self)

    def leaky_relu(self) -> LeakyReLU:
        return LeakyReLU(self)

    def negate(self) -> Negate:
        return Negate(self)

    def exp(self) -> Exp:
        return Exp(self)

    def compute(self) -> None:
        pass

    def get_grad(self) -> None | float:
        pass

    def zero_grad(self) -> None:
        self.grad = 0

    def seed_grad(self, seed: int) -> None:
        self.grad = seed

    def set_value(self, new_value: int | float):
        assert isinstance(new_value, (float, int))
        self.value = float(new_value)

    def trigger_recompute(self) -> None:
        if NodeContext.current_context:
            NodeContext.current_context.recompute()

    def save(self, pickle_instance: Any):
        """
        Saves a `tensorops.node.Node` to a `.pkl` file given a binary file `open()` handle.

        Passing an instance of the file handle would allow for repeated insertion and saving `tensor.node.Node` to a `.pkl` file

        Args:
            pickle_instance (_io.TextIOWrapper): a `.pkl` file handle with write access in binary/
        """

        assert pickle_instance.writable()

        pickle.dump(self, pickle_instance)

    @staticmethod
    def load(
        path: str, limit: int | None = None
    ) -> Node | Generator | itertools.chain[Node]:
        """
        Loads a single `tensorops.node.Node()` or a generator of `tensorops.node.Node()`.

        Args:
            path (str): The file path from which to load the node(s).
            limit (int, optional): Loads the first n items from the pickle file.

        Returns:
            Union[Node, Generator[Node, None, None]]: The loaded node(s).
        """

        def node_generator():
            with open(path, "rb") as f:
                count = 0
                while limit is None or count < limit:
                    try:
                        data = pickle.load(f)
                        if isinstance(data, Node):
                            yield data
                        elif isinstance(data, list) and all(
                            isinstance(item, Node) for item in data
                        ):
                            for item in data:
                                if limit is not None and count >= limit:
                                    return
                                yield item
                                count += 1
                        else:
                            raise ValueError(
                                "All items must be of type `tensorops.node.Node` or a list of `Node`."
                            )
                        count += 1
                    except EOFError:
                        break

        generator = node_generator()
        first_node = next(generator, None)

        if first_node is None:
            return generator

        if limit == 1:
            return first_node

        return itertools.chain([first_node], generator)


class NodeContext:

    """
    `tensorops.NodeContext` manages the operational context for nodes during computation.
    """

    current_context = None

    def __init__(self) -> None:
        self.nodes = []

    def __enter__(self) -> NodeContext:
        self.prev_context = NodeContext.current_context
        NodeContext.current_context = self
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        NodeContext.current_context = self.prev_context

    def add_node(self, node: Node) -> None:
        """
        Creates a node to be added to the computational graph stored in `tensorops.NodeContext.context`
        Args:
            node (tensorops.Node): The node object to be added to the computational graph
        """

        self.nodes.append(node)

    def recompute(self) -> None:
        """
        Recomputes all nodes within the NodeContext.
        """
        forward(self.nodes)

    def weights_enabled(self) -> list[Node]:
        """
        Returns a list of all nodes that are neural network weights

        Returns:
            List[tensorops.node.Node]: list of all nodes that are neural network weights
        """
        return [node for node in self.nodes if node.weight]

    def grad_enabled(self) -> list[Node]:
        """
        Returns a list of all nodes that have gradient tracking enabled.

        Returns:
            List[tensorops.node.Node]: list of all nodes that have gradient tracking enabled.
        """
        return [node for node in self.nodes if node.requires_grad]

    def __repr__(self) -> str:
        return str([node for node in self.nodes])

    def save(self, path: str) -> None:
        """
        Saves the `tensorops.node.NodeContext()` to a `.pkl` file.

        Args:
            path (str): The file path where the `tensorops.node.NodeContext()` should be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> Any:
        """
        Loads a `tensorops.node.NodeContext()` from a `.pkl` file.

        Args:
            path (str): The file path from which to load the `tensorops.node.NodeContext()`.

        Returns:
            tensorops.node.NodeContext(): The loaded context manager to track nodes.
        """
        with open(path, "rb") as f:
            return pickle.load(f)


class Add(Node):
    def __init__(self, node1: Node, node2: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.node2 = node2
        self.parents = [self.node1, self.node2]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = self.node1.value + self.node2.value

    def get_grad(self) -> None:
        self.node1.grad += self.grad * 1
        self.node2.grad += self.grad * 1


class Sub(Node):
    def __init__(self, node1: Node, node2: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.node2 = node2
        self.parents = [self.node1, self.node2]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = self.node1.value - self.node2.value

    def get_grad(self) -> None:
        self.node1.grad += self.grad * 1
        self.node2.grad += self.grad * -1


class Mul(Node):
    def __init__(self, node1: Node, node2: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.node2 = node2
        self.parents = [self.node1, self.node2]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = self.node1.value * self.node2.value

    def get_grad(self) -> None:
        self.node1.grad += self.grad * self.node2.value
        self.node2.grad += self.grad * self.node1.value


class Div(Node):
    def __init__(self, node1: Node, node2: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.node2 = node2
        self.parents = [self.node1, self.node2]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = self.node1.value / self.node2.value

    def get_grad(self) -> None:
        self.node1.grad += self.grad * 1 / self.node2.value
        self.node2.grad += self.grad * -self.node1.value / (self.node2.value**2)


class IntDiv(Node):
    def __init__(self, node1: Node, node2: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.node2 = node2
        self.parents = [self.node1, self.node2]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = self.node1.value / self.node2.value

    def get_grad(self) -> None:
        raise NotImplementedError(
            "Gradient of tensorops.node.IntDiv operator is not implemented!"
        )


class Mod(Node):
    def __init__(self, node1: Node, node2: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.node2 = node2
        self.parents = [self.node1, self.node2]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = self.node1.value % self.node2.value

    def get_grad(self) -> None:
        self.node1.grad += self.grad * 1
        self.node2.grad += self.grad * -(self.node1.value // self.node2.value)


class Power(Node):
    def __init__(self, node1: Node, idx: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.idx = idx
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = self.node1.value**self.idx.value

    def get_grad(self) -> None:
        self.node1.grad += (
            self.grad * self.idx.value * (self.node1.value ** (self.idx.value - 1))
        )


class Sin(Node):
    def __init__(self, node1: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = math.sin(self.node1.value)

    def get_grad(self) -> None:
        self.node1.grad += self.grad * math.cos(self.node1.value)


def sin(node: Node) -> Sin:
    """
    Performs the Sine function to given `tensorops.node.Node`.

    Args:
        node: The input node for Sine.

    Returns:
        tensorops.node.Node.Sin: An instance of Sin with the value set to the output the Sine function.
    """
    return Sin(node)


class Cos(Node):
    def __init__(self, node1: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = math.cos(self.node1.value)

    def get_grad(self) -> None:
        self.node1.grad += self.grad * -math.sin(self.node1.value)


def cos(node: Node) -> Cos:
    """
    Performs the Cosine function to given `tensorops.node.Node`.

    Args:
        node: The input node for Cosine.

    Returns:
        tensorops.node.Node.Cos: An instance of Cos with the value set to the output the Cosine function.
    """
    return Cos(node)


class Tanh(Node):
    def __init__(self, node1: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = math.tanh(self.node1.value)

    def get_grad(self) -> None:
        self.node1.grad += self.grad * (1 - (math.tanh(self.node1.value) ** 2))


def tanh(node: Node) -> Tanh:
    """
    Performs the Tanh (hyperbolic tangent) activation function to given `tensorops.node.Node`.

    Args:
        node: The input node for Tanh.

    Returns:
        tensorops.node.Node.Tanh: An instance of Tanh with the value set to the output the Tanh activation function.
    """
    return Tanh(node)


class ReLU(Node):
    def __init__(self, node1: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = max(0, self.node1.value)

    def get_grad(self) -> None:
        if self.node1.value > 0:
            self.node1.grad += self.grad * 1
        elif self.node1.value < 0:
            self.node1.grad += self.grad * 0


def relu(node: Node) -> ReLU:
    """
    Performs the Sigmoid activation function to given `tensorops.node.Node`.

    Args:
        node: The input node for Sigmoid.

    Returns:
        tensorops.node.Node.Sigmoid: An instance of Sigmoid with the value set to the output the Sigmoid activation function.
    """
    return ReLU(node)


class LeakyReLU(Node):
    def __init__(self, node1: Node, leaky_grad: float = 0.01) -> None:
        super().__init__(0)
        assert isinstance(
            leaky_grad, (float, int)
        ), "leaky_grad parameter can only be float or int"
        self.node1 = node1
        self.parents = [self.node1]
        self.leaky_grad = leaky_grad
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = max(0, self.node1.value) + self.leaky_grad * min(
            0, self.node1.value
        )

    def get_grad(self) -> None:
        if self.node1.value > 0:
            self.node1.grad += self.grad * 1
        elif self.node1.value < 0:
            self.node1.grad += self.grad * self.leaky_grad


def leaky_relu(node: Node, leaky_grad: float = 0.01) -> LeakyReLU:
    """
    Performs the LeakyReLU activation function to given `tensorops.node.Node`.

    Args:
        node (tensorops.node.Node): The input node for LeakyReLU.
        leaky_grad (float): The negative gradient of the activation function. Defaults to 0.01.

    Returns:
        tensorops.node.Node.Sigmoid: An instance of Sigmoid with the value set to the output the Sigmoid activation function.
    """
    return LeakyReLU(node, leaky_grad)


class Ramp(Node):
    def __init__(self, node1: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = math.log(1 + math.exp(-abs(self.node1.value))) + max(
            self.node1.value, 0
        )

    def get_grad(self) -> None:
        self.node1.grad += self.grad * (1.0 / (1.0 + math.exp(-self.node1.value)))


def ramp(node: Node) -> Ramp:
    """
    Performs the Softplus activation function to given `tensorops.node.Node`.

    Args:
        node: The input node for Softplus.

    Returns:
        tensorops.node.Node.Ramp: An instance of Sigmoid with the value set to the output the Softplus activation function.
    """
    return Ramp(node)


class Sigmoid(Node):
    def __init__(self, node1: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = 1.0 / (1.0 + math.exp(-self.node1.value))

    def get_grad(self) -> None:
        sigmoid_value = self.value
        self.node1.grad += self.grad * sigmoid_value * (1.0 - sigmoid_value)


def sigmoid(node: Node) -> Sigmoid:
    """
    Performs the Sigmoid activation function to given `tensorops.node.Node`.

    Args:
        node: The input node for Sigmoid.

    Returns:
        tensorops.node.Node.Sigmoid: An instance of Sigmoid with the value set to the output the Sigmoid activation function.
    """
    return Sigmoid(node)


class Negate(Node):
    def __init__(self, node1: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = self.node1.value * -1

    def get_grad(self) -> None:
        self.node1.grad += self.grad * -1


class Exp(Node):
    def __init__(self, node1: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = math.exp(self.node1.value)

    def get_grad(self) -> None:
        self.node1.grad += self.grad * self.value


def exp(node: Node) -> Exp:
    """
    Performs the exponentiation function to given `tensorops.node.Node`.

    Args:
        node: The input node for exponentiation.

    Returns:
        tensorops.node.Node.Exp: An instance of Exp with the value set to the output the exponentiation function.
    """
    return Exp(node)


class Abs(Node):
    def __init__(self, node1: Node) -> None:
        super().__init__(0)
        self.node1 = node1
        self.parents = [self.node1]
        for parent in self.parents:
            parent.children.append(self)

    def compute(self) -> None:
        self.value = abs(self.node1.value)

    def get_grad(self) -> None:
        if self.node1.value > 0:
            self.node1.grad += self.grad
        elif self.node1.value < 0:
            self.node1.grad += -self.grad


def forward(nodes: list[Node]) -> None:
    """
    Performs the operation of forward pass.

    Args:
        nodes (list[Node]): The list of nodes to operate forward propagation on.
    """
    for node in nodes:
        node.compute()


def backward(nodes: list[Node]) -> None:
    """
    Performs the operation of backpropagation.

    Args:
        nodes: The list of nodes to operate backward propagation on.
    """
    nodes[-1].seed_grad(1)
    for node in reversed(nodes):
        if node.requires_grad:
            node.get_grad()


def zero_grad(nodes: list[Node]) -> None:
    """
    Performs the operation of zeroing gradients to all nodes in the list of `tensorops.node.Nodes`.

    Args:
        nodes: The list of nodes to zero gradients.
    """
    for node in nodes:
        node.zero_grad()
