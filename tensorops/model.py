from __future__ import annotations

import pickle
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Union

from tensorops.loss import Loss
from tensorops.tensor import Tensor, TensorContext


class Model(ABC):
    """
    `tensorops.model.Model` is the abstract base class for a neural network.

    The values of the `tensorops.inputs` and `tensorops.targets` can be changed and accessed by their respective properties, which will trigger recomputation of the graph.

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
        self._prev_layer: Tensor | None = None
        self.seed = seed
        with self.context:
            self.loss_criterion = loss_criterion

    @abstractmethod
    def forward(self, model_inputs: Union[Tensor, Tensor]) -> Union[Tensor, Tensor]:
        """
        Executes a forward pass of the neural network given input.

        Args:
            model_inputs (tensorops.tensor.Tensor): The input for the neural network.

        Returns:
            Tensor: The resulting tensors as output from the calculations of the neural network.
        """

    @abstractmethod
    def calculate_loss(
        self, output: Union[Tensor, Tensor], target: Union[Tensor, Tensor]
    ) -> Union[None, Tensor, Tensor]:
        """
        Calculates the loss between the predicted output from the neural network against the desired output using the cost function set in `tensorops.Model.loss_criterion`.
        Args:
            output (tensorops.tensor.Tensor): The prediction by the neural network to be evaluated against the cost function.
            target (tensorops.tensor.Tensor): The desired output for the neural network given an input.

        Returns:
            tensorops.model.Model.loss (tensorops.tensor.Tensor): The resulting Tensor as an output from the calculations of the neural network.
        """

    def add_layer(
        self,
        num_input_tensors: int,
        num_output_tensors: int,
        activation_function: Callable | None,
    ) -> Layer:
        """
        Adds a new layer to the neural network.

        Args:
            num_input_tensors (int): Number of input tensors to the neural network.
            num_output_tensors (int): Number of output tensors to the neural network.
            activation_function (Callable): A non-linear activation function for the neural network.

        Returns:
            new_layer (tensorops.model.Layer): The newly created layer of the neural network.
        """

        if self.model_layers:
            assert (
                num_input_tensors == self.model_output_layer.num_output_tensors  # type: ignore
            ), f"Layer shapes invalid! Expected current layer shape to be ({self.model_layers[-1].num_output_tensors}, n) from previous layer shape ({self.model_layers[-1].num_input_tensors}, {self.model_layers[-1].num_output_tensors})"
        if self._prev_layer is None:  # initialise with empty tensor
            self._prev_layer = Tensor([0.0] * num_input_tensors)
        new_layer = Layer(
            self.context,
            num_input_tensors,
            num_output_tensors,
            activation_function,
            seed=self.seed,
            layer_input_tensors=self._prev_layer,  # type: ignore
        )
        self._prev_layer = new_layer.layer_output  # type: ignore
        if not self.model_input_layer:
            self.model_input_layer = new_layer
        self.model_layers.append(new_layer)
        self.model_output_layer = self.model_layers[-1]

        self.targets = Tensor([0.0] * self.model_output_layer.num_output_tensors)
        self.weights.append(new_layer.output_weights)
        self.weights.append(new_layer.output_bias)
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
        Performs the backward pass for the current model. Wrapper function for `tensorops.TensorContext.backward()`
        """
        self.context.backward()

    def zero_grad(self) -> None:
        """
        Sets gradients to all tensors within the model to 0.
        """
        for tensor in self.context.ops:
            if isinstance(tensor, Tensor) and tensor.grads is not None:
                tensor.grads = None

    def get_weights(self) -> list[Tensor]:
        """
        Returns the tensors that are weights within the `tensorops.Model` instance. Wrapper function for `tensorops.TensorContext.weights_enabled()`

        Returns:
            list[tensorops.Tensor]: A list of tensors that have `Tensor.weights` set to `True` within tensors present in the `tensorops.Model` instance
        """
        return self.context.weights_enabled()

    def get_gradients(self) -> Tensor:
        """
        Returns the tensors that have gradient tracking enabled within the `tensorops.Model` instance. Wrapper function for `tensorops.TensorContext.get_gradients()`

        Returns:
            list[tensorops.Tensor]: A list of tensors that have `Tensor.requires_grad` set to `True` within tensors present in the `tensorops.Model` instance
        """
        return self.context.grads_enabled()

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
    num_input_tensors (int): Number of neural network inputs in the layer.
    num_output_tensors (int): Number of neurones in the layer.
    activation_function (Optional[Callable]): Non-linear function that determines the activation level of the neurones in the layer based on its input.
    seed (Optional[int, optional]): Optional seed for `random.seed()`.
    layer_input_tensors (Optional[Tensor, optional]): A list of `tensorops.tensor.Tensor` passed as inputs to the layer.
    output_weights (Optional[list[list[float]], optional]): a m × n matrix of `float`, where m is the number of inputs per `model.Activation` and n is the number of `model.Activation` in a layer.
    output_bias (Optional[list[float], optional]): a list of biases for each `tensorops.model.Activation`.
    layer_output (list[tensorops.model.Activation]): Outputs of neural activation from the layer in the form of `tensorops.model.Activation`.
    layer_output_tensors (tensorops.tensor.Tensor): Outputs of neural activation from the layer in the form of `tensorops.tensor.Tensor`.
    """

    def __init__(
        self,
        context: TensorContext,
        num_input_tensors: int,
        num_output_tensors: int,
        activation_function: Callable | None,
        seed: int | None = None,
        layer_input_tensors: Tensor | None = None,
        output_weights: Tensor | None = None,
        output_bias: Tensor | None = None,
    ):
        self.context = context
        self.num_input_tensors = num_input_tensors
        self.num_output_tensors = num_output_tensors
        if layer_input_tensors is not None:
            self.layer_input_tensors = layer_input_tensors
        else:
            self.layer_input_tensors = Tensor([0.0] * num_input_tensors)
        self.activation_function = activation_function
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
        if output_weights is not None:
            assert (
                len(output_weights) == self.num_output_tensors
            ), "Length of output_weights must match num_output_tensors."
            num_weights = self.num_input_tensors * self.num_output_tensors
            assert (
                sum(len(x) for x in output_weights) == num_weights
            ), "Each weight list must match num_input_tensors."
        else:
            output_weights = Tensor(
                [
                    random.uniform(-1, 1)
                    for _ in range(num_input_tensors * num_output_tensors)
                ]
            )
            output_weights = output_weights.reshape(
                (num_input_tensors, num_output_tensors)
            )
        if output_bias is not None:
            assert (
                len(output_bias) == self.num_output_tensors
            ), "Length of output_bias must match num_output_tensors."
        else:
            output_bias = Tensor(
                [random.uniform(-1, 1) for _ in range(self.num_output_tensors)]
            )
        self.output_weights = output_weights
        self.output_bias = output_bias
        input_2d = self.layer_input_tensors.reshape((1, self.num_input_tensors))
        output_2d = input_2d @ self.output_weights + self.output_bias
        output_1d = output_2d.reshape((self.num_output_tensors,))
        self.layer_output = (
            self.activation_function(output_1d)
            if self.activation_function
            else output_1d
        )

    def forward(self, forward_inputs: Tensor):
        """
        Performs a lazy pass for all neurones within a single neural network layer by setting the input values of the neurones.

        Args:
            forward_inputs (tensorops.tensor.Tensor): The input tensors for a single neural network layer.
        Returns:
            Tensor: The list of outputs produced by the neural network layer.
        """
        self.layer_input_tensors.values = forward_inputs.values

    def __repr__(self) -> str:
        return f"""
        {type(self).__name__}(
        num_input_tensors = {self.num_input_tensors},
        num_output_tensors = {self.num_output_tensors},
        weights = {self.output_weights if hasattr(self, "output_weights") else "[Warning]: No weights initialised yet!"},
        bias: {self.output_bias if hasattr(self, "output_bias") else "[Warning]: No bias initialised yet!"},
        activation_function: {self.activation_function.__name__ if self.activation_function else None}
        )
        """

    def __call__(self, forward_inputs: Tensor):
        return self.forward(forward_inputs)
