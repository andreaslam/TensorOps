from abc import ABC, abstractmethod
from tensorops.node import Node, NodeContext, forward, backward, zero_grad
import random
import pickle


class Model(ABC):
    """
    `tensorops.model.Model` is the abstract base class for a neural network.

    The values of the `tensorops.inputs` and `tensorops.targets` can be changed and accessed by their respective properties, which will trigger recomputation of the graph.

    Attributes
    ----------
    context (tensorops.NodeContext): Context manager to keep track and store nodes for forward and backward pass as a computational graph.
    loss_criterion (tensorops.Loss): Cost function of the neural network
    input_nodes (tensorops.Node): Inputs for the model.
    targets (tensorops.Node): Targets for the model.
    path (string): Path to save and load a model.
    """

    def __init__(self, loss_criterion, seed=None):
        self.context = NodeContext()
        self.loss_criterion = loss_criterion
        self.model_layers = []
        self.targets = []
        self.input_layer = None
        self.output_layer = None
        self.seed = seed

    def forward(self, model_inputs):
        """
        Executes a forward pass of the neural network given input.

        Args:
            model_inputs (tensorops.Node): The input for the neural network.

        Returns:
            output_node (tensorops.Node): The resulting node as an output from the calculations of the neural network.
        """
        assert self.input_layer
        assert self.output_layer
        for model_input_nodes, x in zip(self.input_layer.input_nodes, model_inputs):
            model_input_nodes.set_value(x.value)
        model_inputs = self.input_layer.input_nodes
        for layer in self.model_layers:
            model_inputs = layer(model_inputs)
        return self.output_layer.layer_output_nodes

    def calculate_loss(self, output, target):
        """
        Calculates the loss between the predicted output from the neural network against the desired output using the cost function set in `tensorops.Model.loss_criterion`.
        Args:
            output (tensorops.Node): The prediction by the neural network to be evaluated against the cost function.
            target (tensorops.Node): The desired output for the neural network given an input.

        Returns:
            Model.loss (tensorops.Node): The resulting node as an output from the calculations of the neural network.
        """
        for model_target_nodes, training_target, model_output_nodes, output in zip(
            self.targets, target, self.output_layer.layer_output_nodes, output
        ):
            model_output_nodes.set_value(output.value)
            model_target_nodes.set_value(training_target.value)
        return self.loss

    def add_layer(self, num_input_nodes, num_output_nodes, activation_function):
        if self.model_layers:
            assert (
                num_input_nodes == self.output_layer.num_output_nodes
            ), f"Layer shapes invalid! Expected current layer shape to be ({self.model_layers[-1].num_output_nodes}, n) from previous layer shape ({self.model_layers[-1].num_input_nodes}, {self.model_layers[-1].num_output_nodes})"
        new_layer = Layer(
            self.context,
            num_input_nodes,
            num_output_nodes,
            activation_function,
            seed=self.seed,
        )
        if not self.input_layer:
            self.input_layer = new_layer
        self.model_layers.append(new_layer)
        self.output_layer = self.model_layers[-1]

        self.targets = [
            Node(0.0, requires_grad=False)
            for _ in range(self.output_layer.num_output_nodes)
        ]

    def backward(self):
        """
        Performs the backward pass for the current model. Wrapper function for `tensorops.backward()`
        """
        backward(self.context.nodes)

    def zero_grad(self):
        """
        Sets gradients to all nodes within the model to 0. Wrapper function for `tensorops.zero_grad()`
        """
        zero_grad(self.context.nodes)

    def get_weights(self):
        """
        Returns the nodes that are weights within the `tensorops.Model` instance. Wrapper function for `tensorops.NodeContext.weights_enabled()`

        Returns:
            list[tensorops.Node]: A list of nodes that have `Node.weights` set to `True` within nodes present in the `tensorops.Model` instance
        """
        return self.context.weights_enabled()

    def get_gradients(self):
        """
        Returns the nodes that have gradient tracking enabled within the `tensorops.Model` instance. Wrapper function for `tensorops.NodeContext.get_gradients()`

        Returns:
            list[tensorops.Node]: A list of nodes that have `Node.requires_grad` set to `True` within nodes present in the `tensorops.Model` instance
        """
        return self.context.grad_enabled()

    def __call__(self, input_node):
        return self.forward(input_node)

    def __repr__(self):
        if self.model_layers:
            return f"{type(self).__name__}({[layer for layer in self.model_layers]})"

        # if [node for node in self.context.nodes if node.weight]:
        #     return f"{type(self).__name__}(weights={[node for node in self.context.nodes if node.weight]})"
        return "[Warning]: no weights initialised yet"

    def save(self, path):
        """
        Saves the entire model, including layers, nodes, gradients, and optimiser states to a `.pkl` file.

        Args:
            path (str): The file path where the model should be saved.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Loads a model from a `.pkl` file.

        Args:
            path (str): The file path from which to load the model.

        Returns:
            Model: The loaded model.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.context.nodes)


class Layer:
    """
    `tensorops.model.Layer` initialises a layer of a Fully Connected Network (FCN).

    Attributes
    ----------
    context (tensorops.node.NodeContext): Context manager to keep track and store nodes for forward and backward pass as a computational graph.
    weights (list[list[float]]): a m × n matrix of `float`, where m is the number of inputs per `model.Activation` and n is the number of `model.Activation` in a layer.
    bias (list[float]): a list of biases for each `tensorops.model.Activation`.
    seed (Optional[int]): Optional seed for `random.seed()`.
    num_input_nodes (int): Number of neural network inputs in the layer.
    num_output_nodes (int): Number of neurones in the layer.
    activation_function (Optional[Callable]): Non-linear function that determines the activation level of the neurones in the layer based on its input.
    layer_output (list[tensorops.model.Activation]): Outputs of neural activation from the layer in the form of `tensorops.model.Activation`.
    layer_output_nodes (list[tensorops.node.Node]): Outputs of neural activation from the layer in the form of `tensorops.node.Node`.
    """

    def __init__(
        self,
        context,
        num_input_nodes,
        num_output_nodes,
        activation_function,
        output_weights=None,
        output_bias=None,
        seed=None,
    ):
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        self.context = context
        self.num_input_nodes = num_input_nodes
        self.input_nodes = [
            Node(0.0, requires_grad=False, weight=False)
            for _ in range(self.num_input_nodes)
        ]
        self.num_output_nodes = num_output_nodes
        self.activation_function = activation_function

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
                self.num_input_nodes,
                self.activation_function,
                self.context,
                activation_weight,
                activation_bias,
                input_nodes=self.input_nodes,
            )
            for activation_weight, activation_bias in zip(output_weights, output_bias)
        ]
        self.layer_output_nodes = [
            activation.activation_output for activation in self.layer_output
        ]
        self.weights = [activation.weights for activation in self.layer_output]
        self.bias = [activation.bias for activation in self.layer_output]

    def forward(self, forward_inputs):
        """
        Performs a forward pass for all neurones (`tensorops.model.Activation`) within a single neural network layer.

        Args:
            forward_inputs (list[tensorops.node.Node]): The input nodes for a single neural network layer.
        Returns:
            output_nodes (list[tensorops.node.Node]): The list of outputs produced by the neural network layer.
        """

        assert len(forward_inputs) == self.num_input_nodes

        for node, forward_input in zip(self.input_nodes, forward_inputs):
            node.set_value(forward_input.value)

        return [activation(self.input_nodes) for activation in self.layer_output]

    def __repr__(self):
        return f"""
        {type(self).__name__}(
        num_input_nodes = {self.num_input_nodes},
        num_output_nodes = {self.num_output_nodes},
        weights = {self.weights},
        bias: {self.bias},
        activation_function: {self.activation_function.__name__ if self.activation_function else None}
        )
        """

    def __call__(self, forward_inputs):
        return self.forward(forward_inputs)


class Activation:
    """
    `tensorops.model.Activation` mimics a biological neurone, taking a sum of multiplications between each weight and input, and adding a bias (a constant) on top of the sum and applying activation function at the end.

    Mathematically, a neurone is defined as:
        f(sum(x_i, w_i) + c)
    where:
        - f() is the non-linear activation function.
        - x_i, w_i are the ith input and weight in a list containing all inputs and weights for the neurone respectively.
        - c is a constant added to the function.

    Attributes
    ----------
    context (tensorops.node.NodeContext): Context manager to keep track and store nodes for forward and backward pass as a computational graph.
    num_input_nodes (int): Number of neural network inputs for the neurone.
    seed (Optional[int]): Optional seed for `random.seed()`.
    input_nodes (list[tensorops.node.Node]): Inputs for the neurone.
    weights (list[float]): Weights for each input.
    bias (float): Constant added to the function.
    activation_function (Optional[Callable]): Non-linear function that determines the activation level of the neurone based on its input.
    activation_output (tensorops.node.Node): Output node of neural activation.
    """

    def __init__(
        self,
        num_input_nodes,
        activation_function,
        context,
        weights=None,
        bias=None,
        seed=None,
        input_nodes=None,
    ):
        self.num_input_nodes = num_input_nodes
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        self.context = context
        with self.context:
            if input_nodes:
                assert len(input_nodes) == num_input_nodes
                self.input_nodes = input_nodes
            else:
                self.input_nodes = [
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
            for weight, input_node in zip(self.weights, self.input_nodes):
                output_node += weight * input_node

            if self.activation_function:
                self.activation_output = self.activation_function(
                    output_node + self.bias
                )
            else:
                self.activation_output = output_node + self.bias

    def forward(self, forward_inputs):
        """
        Performs an activation step for a single neurone.

        Args:
            forward_inputs (list[tensorops.node.Node]): The input nodes for a single neurone.
        """

        assert len(forward_inputs) == self.num_input_nodes
        for node, forward_input in zip(self.input_nodes, forward_inputs):
            node.set_value(forward_input.value)
        forward(self.context.nodes)
        return self.activation_output

    def __call__(self, forward_inputs):
        return self.forward(forward_inputs)

    def __repr__(self):
        weights_and_inputs_string = ", ".join(
            f"(weight_{num}: {weight}, input_{num}: {node})"
            for num, (weight, node) in enumerate(zip(self.weights, self.input_nodes))
        )

        return f"""
        {type(self).__name__}(
        {weights_and_inputs_string},
        bias: {self.bias},
        activation_function: {self.activation_function.__name__}
        )
        """
