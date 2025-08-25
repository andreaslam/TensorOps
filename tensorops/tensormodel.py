from __future__ import annotations
from typing import Any, Callable, Union

from tensorops.loss import Loss
from tensorops.tensor import TensorContext, Tensor
import random
import pickle
from abc import ABC, abstractmethod


class Model(ABC):
    """
    `tensorops.model.Model` is the abstract base class for a neural network.

    The valuess of the `tensorops.inputs` and `tensorops.targets` can be changed and accessed by their respective properties, which will trigger recomputation of the graph.

    Attributes
    ----------
    loss_criterion (tensorops.loss.Loss): Cost function of the neural network.
    context (tensorops.tensor.TensorContext): Context manager to keep track and store tensors for forward and backward pass as a computational graph.
    model_layers (list[tensorops.model.Layers]): A list containing the all the layers of the neural network.
    targets (tensorops.tensor.Tensor): Targets for the model.
    weights (tensorops.tensor.Tensor): Contains all the weights of the neural network.
    model_input_layer (Union[list[tensorops.model.Layer], None]): The input layer of the neural network. Returns `None` if it is not initialised.
    model_output_layer (Union[list[tensorops.model.Layer], None]): The output layer of the neural network. Returns `None` if it is not initialised.
    loss (Optional[tensorops.tensor.Tensor, None]): The output of the loss function. `None` if the loss function/calculation is not defined.
    seed (Optional[int, None], optional): Seed for generating random weights using the `random` library.
    path (string): Path to save and load a model.
    input_tensors (tensorops.tensor.Tensor): Inputs for the model.
    """

    def __init__(self, loss_criterion: Loss, seed: int | None = None) -> None:
        self.context = TensorContext()
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
    def forward(self, model_inputs: Tensor) -> Tensor:
        """
        Executes a forward pass of the neural network given input.

        Args:
            model_inputs (tensorops.tensor.Tensor): The input for the neural network.

        Returns:
            tensorops.tensor.Tensor: The resulting tensors as output from the calculations of the neural network.
        """

    @abstractmethod
    def calculate_loss(self, output: Tensor, target: Tensor) -> Union[None, Tensor]:
        """
        Calculates the loss between the predicted output from the neural network against the desired output using the cost function set in `tensorops.Model.loss_criterion`.
        Args:
            output (tensorops.tensor.Tensor): The prediction by the neural network to be evaluated against the cost function.
            target (tensorops.tensor.Tensor): The desired output for the neural network given an input.

        Returns:
            tensorops.model.Model.loss (tensorops.tensor.Tensor): The resulting tensor as an output from the calculations of the neural network.
        """

    def add_layer(
        self,
        num_input_values: int,
        num_output_tensors: int,
        activation_function: Callable | None,
    ) -> Layer:
        """
        Adds a new layer to the neural network.

        Args:
            num_input_values (int): Number of input tensors to the neural network.
            num_output_tensors (int): Number of output tensors to the neural network.
            activation_function (Callable): A non-linear activation function for the neural network.

        Returns:
            new_layer (tensorops.model.Layer): The newly created layer of the neural network.
        """

        if self.model_layers:
            assert (
                num_input_values == self.model_output_layer.num_output_tensors  # type: ignore
            ), f"Layer shapes invalid! Expected current layer shape to be ({self.model_layers[-1].num_output_tensors}, n) from previous layer shape ({self.model_layers[-1].num_input_values}, {self.model_layers[-1].num_output_tensors})"
        new_layer = Layer(
            self.context,
            num_input_values,
            num_output_tensors,
            activation_function,
            seed=self.seed,
            layer_input_tensors=self._prev_layer,  # type: ignore
        )
        self._prev_layer = new_layer.layer_output_tensors  # type: ignore
        if not self.model_input_layer:
            self.model_input_layer = new_layer
        self.model_layers.append(new_layer)
        self.model_output_layer = self.model_layers[-1]

        self.targets = Tensor(
            [0.0 for _ in range(self.model_output_layer.num_output_tensors)],
            requires_grad=False,
        )
        # type: ignore
        self.weights = self.context.weights_enabled()
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
        self.context.backward()

    def zero_grad(self) -> None:
        """
        Sets gradients to all tensors within the model to 0. Wrapper function for `tensorops.zero_grad()`
        """
        for weight in self.weights:
            weight.grads.values = [0.0] * len(weight.grads.values)

    def save(self, path: str) -> None:
        """
        Saves the entire model, including layers, tensors, gradients, and optimiser states to a `.pkl` file.

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
        return len(self.context.ops)

    def __call__(self, input_tensor: Tensor | Tensor) -> Tensor | Tensor:
        return self.forward(input_tensor)

    def __repr__(self) -> str:
        if self.model_layers:
            return f"{type(self).__name__}({[layer for layer in self.model_layers]})"
        return "[Warning]: no weights initialised yet"


class Layer:
    """
    `tensorops.model.Layer` initialises a layer of a Fully Connected Network (FCN).

    Attributes
    ----------
    context (tensorops.tensor.TensorContext): Context manager to keep track and store tensors for forward and backward pass as a computational graph.
    num_input_values (int): Number of neural network inputs in the layer.
    num_output_tensors (int): Number of neurones in the layer.
    activation_function (Optional[Callable]): Non-linear function that determines the activation level of the neurones in the layer based on its input.
    seed (Optional[int, optional]): Optional seed for `random.seed()`.
    layer_input_tensors (Optional[tensorops.tensor.Tensor, optional]): A list of `tensorops.tensor.Tensor` passed as inputs to the layer.
    output_weights (Optional[list[tensorops.tensor.Tensor], optional]): a m × n matrix of `float`, where m is the number of inputs per `model.Activation` and n is the number of `model.Activation` in a layer.
    output_bias (Optional[tensorops.tensor.Tensor, list[float], optional]): a list of biases for each `tensorops.model.Activation`.
    layer_output (list[tensorops.model.Activation]): Outputs of neural activation from the layer in the form of `tensorops.model.Activation`.
    layer_output_tensors (tensorops.tensor.Tensor): Outputs of neural activation from the layer in the form of `tensorops.tensor.Tensor`.
    """

    def __init__(
        self,
        context: TensorContext,
        num_input_values: int,
        num_output_tensors: int,
        activation_function: Callable | None,
        seed: int | None = None,
        layer_input_tensors: Tensor | None = None,
        output_weights: list[Tensor] | None = None,
        output_bias: list[float | Tensor] | None = None,
    ):
        self.context = context
        self.num_input_values = num_input_values
        self.num_output_tensors = num_output_tensors
        if layer_input_tensors:
            self.layer_input_tensors = layer_input_tensors
        else:
            self.layer_input_tensors = Tensor(
                [0.0 for _ in range(self.num_input_values)],
                requires_grad=False,
                weight=False,
            )
        self.activation_function = activation_function
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        if output_weights:
            assert (
                len(output_weights) == self.num_output_tensors
            ), "Length of output_weights must match num_output_tensors."
            num_weights = self.num_input_values * self.num_output_tensors
            assert (
                sum(len(x) for x in output_weights) == num_weights
            ), "Each weight list must match num_input_values."
        else:
            output_weights = [
                Tensor(
                    [random.uniform(-1, 1) for _ in range(num_input_values)],
                    requires_grad=True,
                    weight=True,
                )
                for _ in range(num_output_tensors)
            ]
        if output_bias:
            assert (
                len(output_bias) == self.num_output_tensors
            ), "Length of output_bias must match num_output_tensors."
        else:
            output_bias = [
                Tensor(random.uniform(-1, 1), requires_grad=True)
                for _ in range(self.num_output_tensors)
            ]
        self.layer_output = [
            Activation(
                self.context,
                self.num_input_values,
                self.activation_function,
                activation_weight,
                activation_bias,
                activation_input_tensors=self.layer_input_tensors,
            )
            for activation_weight, activation_bias in zip(output_weights, output_bias)
        ]
        self.layer_output_tensors = [
            activation.activation_output for activation in self.layer_output
        ]
        self.weights = [activation.weights for activation in self.layer_output]
        self.bias = [activation.bias for activation in self.layer_output]

    def forward(self, forward_inputs: Tensor) -> Tensor:
        """
        Performs a forward pass for all neurones (`tensorops.model.Activation`) within a single neural network layer.

        Args:
            forward_inputs (tensorops.tensor.Tensor): The input tensors for a single neural network layer.
        Returns:
            tensorops.tensor.Tensor: The list of outputs produced by the neural network layer.
        """

        assert (
            len(forward_inputs) == self.num_input_values
        ), f"Inputs length {len(forward_inputs)} != number of input tensors of layer {self.num_input_values}"

        for tensor, forward_input in zip(self.layer_input_tensors, forward_inputs):
            tensor.set_values(forward_input.values)

        return [
            activation(self.layer_input_tensors) for activation in self.layer_output
        ]

    def __repr__(self) -> str:
        return f"""
        {type(self).__name__}(
        num_input_values = {self.num_input_values},
        num_output_tensors = {self.num_output_tensors},
        weights = {self.weights if self.weights else "[Warning]: No weights initialised yet!"},
        bias: {self.bias if self.bias else "[Warning]: No bias initialised yet!"},
        activation_function: {self.activation_function.__name__ if self.activation_function else None}
        )
        """

    def __call__(self, forward_inputs: Tensor) -> Tensor:
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
    context (tensorops.tensor.TensorContext): Context manager to keep track and store tensors for forward and backward pass as a computational graph.
    num_input_values (int): Number of neural network inputs for the neurone.
    activation_function (Optional[Callable]): Non-linear function that determines the activation level of the neurone based on its input.
    weights (tensorops.tensor.Tensor): Weights for each input.
    bias (Union[float, Tensor]): Constant added to the function.
    seed (Optional[int]): Optional seed for `random.seed()`.
    activation_input_tensors (tensorops.tensor.Tensor): Inputs for the neurone.
    activation_output (tensorops.tensor.Tensor): Output tensor of neural activation.
    """

    def __init__(
        self,
        context: TensorContext,
        num_input_values: int,
        activation_function: Callable | None,
        weights: Tensor | None = None,
        bias: float | Tensor | None = None,
        seed: int | None = None,
        activation_input_tensors: Tensor | None = None,
    ):
        self.num_input_values = num_input_values
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        self.context = context
        with self.context:
            if activation_input_tensors:
                assert len(activation_input_tensors) == num_input_values
                self.activation_input_tensors = activation_input_tensors
            else:
                self.activation_input_tensors = Tensor(
                    [0.0] * self.num_input_values, requires_grad=False, weight=False
                )
            self.activation_function = activation_function
            if weights:
                assert len(weights) == self.num_input_values
                self.weights = weights
            else:
                self.weights = Tensor(
                    [random.uniform(-1, 1) for _ in range(self.num_input_values)],
                    requires_grad=True,
                    weight=True,
                )

            if bias:
                self.bias = (
                    bias
                    if isinstance(bias, Tensor)
                    else Tensor(bias, requires_grad=True, weight=True)
                )
            else:
                self.bias = Tensor(
                    random.uniform(-1, 1), requires_grad=True, weight=True
                )

            self.activation_output = (
                self.activation_function(
                    self.weights * self.activation_input_tensors + self.bias
                )
                if self.activation_function
                else self.weights * self.activation_input_tensors + self.bias
            )

    def forward(self, forward_inputs: Tensor) -> Tensor:
        """
        Performs an activation step for a single neurone.

        Args:
            forward_inputs (tensorops.tensor.Tensor): The input tensors for a single neurone.
        Returns:
            tensorops.tensor.Tensor: The output of the neuronal activation.
        """

        assert len(forward_inputs) == self.num_input_values and forward_inputs.values
        self.activation_input_tensors.values = forward_inputs.values
        self.context.forward()
        return self.activation_output

    def __call__(self, forward_inputs: Tensor) -> Tensor:
        return self.forward(forward_inputs)

    def __repr__(self) -> str:
        weights_and_inputs_string = ", ".join(
            f"(weight_{num}: {weight}, input_{num}: {tensor})"
            for num, (weight, tensor) in enumerate(
                zip(self.weights, self.activation_input_tensors)
            )
        )

        return f"""
        {type(self).__name__}(
        {weights_and_inputs_string},
        bias: {self.bias},
        activation_function: {self.activation_function.__name__ if self.activation_function else None}
        )
        """
