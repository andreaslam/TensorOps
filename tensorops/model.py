from __future__ import annotations
from typing import Any, Callable, Union

from tensorops.loss import Loss
from tensorops.node import Node, NodeContext, forward, backward, zero_grad
import random
import pickle
from abc import ABC, abstractmethod


class Model(ABC):
    """
    `tensorops.model.Model` is the abstract base class for a neural network.

    The values of the `tensorops.inputs` and `tensorops.targets` can be changed and accessed by their respective properties, which will trigger recomputation of the graph.

    Attributes
    ----------
    loss_criterion (tensorops.loss.Loss): Cost function of the neural network.
    context (tensorops.node.NodeContext): Context manager to keep track and store nodes for forward and backward pass as a computational graph.
    model_layers (list[tensorops.model.Layers]): A list containing the all the layers of the neural network.
    targets (tensorops.node.Node): Targets for the model.
    weights (list[tensorops.node.Node]): Contains all the weights of the neural network.
    model_input_layer (Union[list[tensorops.model.Layer], None]): The input layer of the neural network. Returns `None` if it is not initialised.
    model_output_layer (Union[list[tensorops.model.Layer], None]): The output layer of the neural network. Returns `None` if it is not initialised.
    loss (Optional[tensorops.node.Node, None]): The output of the loss function. `None` if the loss function/calculation is not defined.
    seed (Optional[int, None], optional): Seed for generating random weights using the `random` library.
    path (string): Path to save and load a model.
    input_nodes (tensorops.node.Node): Inputs for the model.
    """

    def __init__(self, loss_criterion: Loss, seed: int | None = None) -> None:
        self.context = NodeContext()
        self.model_layers = []
        self.targets = []
        self.weights = []
        self.model_input_layer: Union[Layer, None] = None
        self.model_output_layer: Union[Layer, None] = None
        self.loss = None
        self._prev_layer: list[Layer] | None = None
        self.seed = seed
        with self.context:
            self.loss_criterion = loss_criterion

    @abstractmethod
    def forward(self, model_inputs: Union[list[Node], Node]) -> Union[list[Node], Node]:
        """
        Executes a forward pass of the neural network given input.

        Args:
            model_inputs (list[tensorops.node.Node]): The input for the neural network.

        Returns:
            list[tensorops.node.Node]: The resulting nodes as output from the calculations of the neural network.
        """

    @abstractmethod
    def calculate_loss(
        self, output: Union[list[Node], Node], target: Union[list[Node], Node]
    ) -> Union[None, Node, list[Node]]:
        """
        Calculates the loss between the predicted output from the neural network against the desired output using the cost function set in `tensorops.Model.loss_criterion`.
        Args:
            output (tensorops.node.Node): The prediction by the neural network to be evaluated against the cost function.
            target (tensorops.node.Node): The desired output for the neural network given an input.

        Returns:
            tensorops.model.Model.loss (tensorops.node.Node): The resulting node as an output from the calculations of the neural network.
        """

    def add_layer(
        self,
        num_input_nodes: int,
        num_output_nodes: int,
        activation_function: Callable | None,
    ) -> Layer:
        """
        Adds a new layer to the neural network.

        Args:
            num_input_nodes (int): Number of input nodes to the neural network.
            num_output_nodes (int): Number of output nodes to the neural network.
            activation_function (Callable): A non-linear activation function for the neural network.

        Returns:
            new_layer (tensorops.model.Layer): The newly created layer of the neural network.
        """

        if self.model_layers:
            assert (
                num_input_nodes == self.model_output_layer.num_output_nodes  # type: ignore
            ), f"Layer shapes invalid! Expected current layer shape to be ({self.model_layers[-1].num_output_nodes}, n) from previous layer shape ({self.model_layers[-1].num_input_nodes}, {self.model_layers[-1].num_output_nodes})"
        new_layer = Layer(
            self.context,
            num_input_nodes,
            num_output_nodes,
            activation_function,
            seed=self.seed,
            layer_input_nodes=self._prev_layer,  # type: ignore
        )
        self._prev_layer = new_layer.layer_output_nodes  # type: ignore
        if not self.model_input_layer:
            self.model_input_layer = new_layer
        self.model_layers.append(new_layer)
        self.model_output_layer = self.model_layers[-1]

        self.targets = [
            Node(0.0, requires_grad=False)
            for _ in range(self.model_output_layer.num_output_nodes)  # type: ignore
        ]
        self.weights = self.get_weights()
        return new_layer

    def eval(self) -> None:
        """
        Disables gradient tracking in `tensorops.Model`.
        """
        for weight in self.weights:
            weight.requires_grad = False

    def train(self) -> None:
        """
        Enables gradient tracking in `tensorops.Model`.
        """
        for weight in self.weights:
            weight.requires_grad = True

    def backward(self) -> None:
        """
        Performs the backward pass for the current model. Wrapper function for `tensorops.backward()`
        """
        backward(self.context.nodes)

    def zero_grad(self) -> None:
        """
        Sets gradients to all nodes within the model to 0. Wrapper function for `tensorops.zero_grad()`
        """
        zero_grad(self.context.nodes)

    def get_weights(self) -> list[Node]:
        """
        Returns the nodes that are weights within the `tensorops.Model` instance. Wrapper function for `tensorops.NodeContext.weights_enabled()`

        Returns:
            list[tensorops.Node]: A list of nodes that have `Node.weights` set to `True` within nodes present in the `tensorops.Model` instance
        """
        return self.context.weights_enabled()

    def get_gradients(self) -> list[Node]:
        """
        Returns the nodes that have gradient tracking enabled within the `tensorops.Model` instance. Wrapper function for `tensorops.NodeContext.get_gradients()`

        Returns:
            list[tensorops.Node]: A list of nodes that have `Node.requires_grad` set to `True` within nodes present in the `tensorops.Model` instance
        """
        return self.context.grad_enabled()

    def save(self, path: str) -> None:
        """
        Saves the entire model, including layers, nodes, gradients, and optimiser states to a `.pkl` file.

        Args:
            path (str): The file path where the model should be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> Any:
        """
        Loads a model from a `.pkl` file.

        Args:
            path (str): The file path from which to load the model.

        Returns:
            Model: The loaded model.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __len__(self) -> int:
        return len(self.context.nodes)

    def __call__(self, input_node: list[Node] | Node) -> list[Node] | Node:
        return self.forward(input_node)

    def __repr__(self) -> str:
        if self.model_layers:
            return f"{type(self).__name__}({[layer for layer in self.model_layers]})"
        return "[Warning]: no weights initialised yet"


class Layer:
    """
    `tensorops.model.Layer` initialises a layer of a Fully Connected Network (FCN).

    Attributes
    ----------
    context (tensorops.node.NodeContext): Context manager to keep track and store nodes for forward and backward pass as a computational graph.
    num_input_nodes (int): Number of neural network inputs in the layer.
    num_output_nodes (int): Number of neurones in the layer.
    activation_function (Optional[Callable]): Non-linear function that determines the activation level of the neurones in the layer based on its input.
    seed (Optional[int, optional]): Optional seed for `random.seed()`.
    layer_input_nodes (Optional[list[tensorops.node.Node], optional]): A list of `tensorops.node.Node` passed as inputs to the layer.
    output_weights (Optional[list[list[float]], optional]): a m × n matrix of `float`, where m is the number of inputs per `model.Activation` and n is the number of `model.Activation` in a layer.
    output_bias (Optional[list[float], optional]): a list of biases for each `tensorops.model.Activation`.
    layer_output (list[tensorops.model.Activation]): Outputs of neural activation from the layer in the form of `tensorops.model.Activation`.
    layer_output_nodes (list[tensorops.node.Node]): Outputs of neural activation from the layer in the form of `tensorops.node.Node`.
    """

    def __init__(
        self,
        context: NodeContext,
        num_input_nodes: int,
        num_output_nodes: int,
        activation_function: Callable | None,
        seed: int | None = None,
        layer_input_nodes: list[Node] | None = None,
        output_weights: list[list[float]] | None = None,
        output_bias: list[float] | None = None,
    ):
        self.context = context
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        if layer_input_nodes:
            self.layer_input_nodes = layer_input_nodes
        else:
            self.layer_input_nodes = [
                Node(0.0, requires_grad=False, weight=False)
                for _ in range(self.num_input_nodes)
            ]
        self.activation_function = activation_function
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        if output_weights:
            assert (
                len(output_weights) == self.num_output_nodes
            ), "Length of output_weights must match num_output_nodes."
            num_weights = self.num_input_nodes * self.num_output_nodes
            assert (
                sum(len(x) for x in output_weights) == num_weights
            ), "Each weight list must match num_input_nodes."
        else:
            output_weights = [
                [random.uniform(-1, 1) for _ in range(num_input_nodes)]
                for _ in range(num_output_nodes)
            ]
        if output_bias:
            assert (
                len(output_bias) == self.num_output_nodes
            ), "Length of output_bias must match num_output_nodes."
        else:
            output_bias = [random.uniform(-1, 1) for _ in range(self.num_output_nodes)]
        self.layer_output = [
            Activation(
                self.context,
                self.num_input_nodes,
                self.activation_function,
                activation_weight,
                activation_bias,
                activation_input_nodes=self.layer_input_nodes,
            )
            for activation_weight, activation_bias in zip(output_weights, output_bias)
        ]
        self.layer_output_nodes = [
            activation.activation_output for activation in self.layer_output
        ]
        self.weights = [activation.weights for activation in self.layer_output]
        self.bias = [activation.bias for activation in self.layer_output]

    def forward(self, forward_inputs: list[Node]) -> list[Node]:
        """
        Performs a forward pass for all neurones (`tensorops.model.Activation`) within a single neural network layer.

        Args:
            forward_inputs (list[tensorops.node.Node]): The input nodes for a single neural network layer.
        Returns:
            list[tensorops.node.Node]: The list of outputs produced by the neural network layer.
        """

        assert (
            len(forward_inputs) == self.num_input_nodes
        ), f"Inputs length {len(forward_inputs)} != number of input nodes of layer {self.num_input_nodes}"

        for node, forward_input in zip(self.layer_input_nodes, forward_inputs):
            node.set_value(forward_input.value)

        return [activation(self.layer_input_nodes) for activation in self.layer_output]

    def __repr__(self) -> str:
        return f"""
        {type(self).__name__}(
        num_input_nodes = {self.num_input_nodes},
        num_output_nodes = {self.num_output_nodes},
        weights = {self.weights if self.weights else "[Warning]: No weights initialised yet!"},
        bias: {self.bias if self.bias else "[Warning]: No bias initialised yet!"},
        activation_function: {self.activation_function.__name__ if self.activation_function else None}
        )
        """

    def __call__(self, forward_inputs: list[Node]) -> list[Node]:
        return self.forward(forward_inputs)


class Activation:
    """
    `tensorops.model.Activation` mimics a biological neurone, taking a sum of multiplications between each weight and input, and adding a bias (a constant) on top of the sum and applying activation function at the end.

    Mathematically, a neurone is defined as:
        f(sum(x_i, w_i) + c)
    where:
        - f(x) is the non-linear activation function.
        - x_i, w_i are the ith input and weight in a list containing all inputs and weights for the neurone respectively.
        - c is a constant added to the function.

    Attributes
    ----------
    context (tensorops.node.NodeContext): Context manager to keep track and store nodes for forward and backward pass as a computational graph.
    num_input_nodes (int): Number of neural network inputs for the neurone.
    seed (Optional[int]): Optional seed for `random.seed()`.
    activation_input_nodes (list[tensorops.node.Node]): Inputs for the neurone.
    weights (list[float]): Weights for each input.
    bias (float): Constant added to the function.
    activation_function (Optional[Callable]): Non-linear function that determines the activation level of the neurone based on its input.
    activation_output (tensorops.node.Node): Output node of neural activation.
    """

    def __init__(
        self,
        context: NodeContext,
        num_input_nodes: int,
        activation_function: Callable | None,
        weights: list[float] | None = None,
        bias: float | None = None,
        seed: int | None = None,
        activation_input_nodes: list[Node] | None = None,
    ):
        self.num_input_nodes = num_input_nodes
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        self.context = context
        with self.context:
            if activation_input_nodes:
                assert len(activation_input_nodes) == num_input_nodes
                self.activation_input_nodes = activation_input_nodes
            else:
                self.activation_input_nodes = [
                    Node(0.0, requires_grad=False, weight=False)
                    for _ in range(self.num_input_nodes)
                ]
            self.activation_function = activation_function
            if weights:
                assert len(weights) == self.num_input_nodes
                self.weights = [
                    Node(weight, requires_grad=True, weight=True) for weight in weights
                ]
            else:
                self.weights = [
                    Node(random.uniform(-1, 1), requires_grad=True, weight=True)
                    for _ in range(self.num_input_nodes)
                ]

            if bias:
                self.bias = Node(bias, requires_grad=True, weight=True)
            else:
                self.bias = Node(random.uniform(-1, 1), requires_grad=True, weight=True)

            output_node = Node(0.0, requires_grad=True, weight=False)
            for weight, input_node in zip(self.weights, self.activation_input_nodes):
                output_node += weight * input_node

            if self.activation_function:
                self.activation_output = self.activation_function(
                    output_node + self.bias
                )
            else:
                self.activation_output = output_node + self.bias

    def forward(self, forward_inputs: list[Node]) -> Node:
        """
        Performs an activation step for a single neurone.

        Args:
            forward_inputs (list[tensorops.node.Node]): The input nodes for a single neurone.
        Returns:
            tensorops.node.Node: The output of the neuronal activation.
        """

        assert len(forward_inputs) == self.num_input_nodes
        for node, forward_input in zip(self.activation_input_nodes, forward_inputs):
            node.set_value(forward_input.value)
        forward(self.context.nodes)
        return self.activation_output

    def __call__(self, forward_inputs: list[Node]) -> Node:
        return self.forward(forward_inputs)

    def __repr__(self) -> str:
        weights_and_inputs_string = ", ".join(
            f"(weight_{num}: {weight}, input_{num}: {node})"
            for num, (weight, node) in enumerate(
                zip(self.weights, self.activation_input_nodes)
            )
        )

        return f"""
        {type(self).__name__}(
        {weights_and_inputs_string},
        bias: {self.bias},
        activation_function: {self.activation_function.__name__ if self.activation_function else None}
        )
        """
