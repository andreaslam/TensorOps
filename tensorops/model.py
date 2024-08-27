from abc import ABC, abstractmethod
from tensorops.node import Node, NodeContext, Sigmoid, backward, zero_grad


class Model(ABC):
    """
    `tensorops.Model` is the abstract base class for a neural network.

    The values of the `tensorops.inputs` and `tensorops.targets` can be changed and accessed by their respective properties, which will trigger recomuptation of the graph.

    Attributes
    ----------
    context (tensorops.NodeContext): Context manager to keep track and store nodes for forward and backward pass as a computational graph.
    loss_criterion (tensorops.Loss): Cost function of the neural network
    inputs (tensorops.Node): Inputs for the model.
    targets (tensorops.Node): Targets for the model.
    """

    def __init__(self, loss_criterion):
        self.context = NodeContext()
        self.loss_criterion = loss_criterion
        with self.context:
            self._inputs = Node(0.0, requires_grad=False)
            self._targets = Node(0.0, requires_grad=False)

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs.set_value(value)

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets.set_value(value)

    @abstractmethod
    def forward(self, input_node):
        """
        Executes a forward pass of the neural network given input.

        Args:
            input_node (tensorops.Node): The input for the neural network.

        Returns:
            output_node (tensorops.Node): The resulting node as an output from the calculations of the neural network.
        """

        ...

    @abstractmethod
    def calculate_loss(self, output, target):
        """
        Calulates the loss between the predicted output from the neural network against the desired output using the cost function set in tensorops.Model.loss_criterion.
        Args:
            output (tensorops.Node): The prediction by the neural network to be evaluated against the cost function.
            target (tensorops.Node): The desired output for the neural network given an input.

        Returns:
            Model.loss (tensorops.Node): The resulting node as an output from the calculations of the neural network.
        """
        ...

    def add_node(self, value, requires_grad=True, weight=False):
        """
        Creates a node to be added to the computational graph stored in `tensorops.Model.context`
        Args:
            value (float): The value of the node to be created.
            requires_grad (bool): Whether the node requires gradient tracking.
            weight (bool): Whether the node is a neural network weight.

        Returns:
            Model.output_node (tensorops.Node): that has been appended to computational graph.
        """

        with self.context:
            return Node(value, requires_grad, weight)

    def backward(self):
        backward(self.context.nodes)

    def zero_grad(self):
        zero_grad(self.context.nodes)

    def get_weights(self):
        return self.context.weights_enabled()

    def get_gradients(self):
        return self.context.grad_enabled()

    def __call__(self, input_node):
        return self.forward(input_node)

    def __repr__(self):
        if self.context.nodes:
            return f"{type(self).__name__}(weights={[node for node in self.context.nodes if node.weight]})"
        return "[Warning]: no weights initialised yet"


def sigmoid(node):
    return Sigmoid(node)
