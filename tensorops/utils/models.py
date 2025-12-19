from typing import Callable
from tensorops.model import Model
from tensorops.tensor import Tensor, TensorContext
from tensorops.loss import Loss


class SimpleSequentialModel(Model):
    """
    # A simple `tensorops.model.Model` subclass used for demonstration purposes to create a `tensorops.model.Model` without a `tensorops.Model.Layer` class.

    It has a network architecture of y=mx+c. Additional activation functions could be configured.

    Attributes
    ----------
    loss_criterion (tensorops.loss.Loss): Cost function of the neural network.
    activation_function (Optional[Callable]): Optional activation function for the neural network. Defaults to None.
    """

    def __init__(
        self, loss_criterion: Loss, activation_function: Callable | None = None
    ) -> None:
        super().__init__(loss_criterion)
        self.context = TensorContext()
        self.loss_criterion = loss_criterion
        self.model_layers = []
        with self.context:
            self.activation_function = activation_function
            self.input_Tensor = Tensor(0.0, requires_grad=False)
            if self.activation_function:
                self.output_Tensor = self.activation_function(
                    Tensor(0.0, requires_grad=False)
                )
            else:
                self.output_Tensor = Tensor(0.0, requires_grad=False)
            self.targets = Tensor(0.0, requires_grad=False)
            self.loss = loss_criterion.loss(self.targets, self.output_Tensor)

    def forward(self, model_inputs: Tensor) -> Tensor:  # type: ignore
        assert self.output_Tensor, "Output behaviour not defined"
        with self.context:
            self.input_Tensor.set_value(model_inputs.value)
            forward(self.context.Tensors)
            return self.output_Tensor

    def calculate_loss(self, output: Tensor, target: Tensor) -> Tensor:  # type: ignore
        assert self.output_Tensor, "Output behaviour not defined"
        with self.context:
            self.output_Tensor.set_value(output.value)
            self.targets.set_value(target.value)
            self.input_Tensor.trigger_recompute()
            return self.loss

    def __repr__(self) -> str:
        if [Tensor for Tensor in self.context.Tensors if Tensor.weight]:
            return f"{type(self).__name__}(weights={[Tensor for Tensor in self.context.Tensors if Tensor.weight]})"
        return "[Warning]: no weights initialised yet"


class SequentialModel(Model):
    """
    A general-purpose `tensorops.model.Model` subclass used to create a customisable `tensorops.model.Model`.

    This assumes that each layer is used sequentially without additional set-up for the forward pass.
    """

    def __init__(self, loss_criterion: Loss, seed=None) -> None:
        super().__init__(loss_criterion, seed)

    def forward(self, model_inputs: list[Tensor]) -> list[Tensor]:  # type: ignore
        assert (
            self.model_input_layer and self.model_input_layer.layer_input_Tensors
        ), f"{type(self).__name__}.input_layer not defined!"
        assert (
            self.model_output_layer and self.model_output_layer.layer_output_Tensors
        ), f"{type(self).__name__}.output_layer not defined!"
        assert len(model_inputs) == len(
            self.model_input_layer.layer_input_Tensors
        ), f"Inputs length {len(model_inputs)} != number of input Tensors of model {len(self.model_input_layer.layer_input_Tensors)}"
        with self.context:
            for layer in self.model_layers:
                model_inputs = layer(model_inputs)
        return model_inputs

    def calculate_loss(self, output: list[Tensor], target: list[Tensor]) -> Tensor:  # type: ignore
        """
        Calculates the loss between the predicted output from the neural network against the desired output using the cost function set in `tensorops.Model.loss_criterion`.
        Args:
            output (list[tensorops.Tensor.Tensor]): The prediction by the neural network to be evaluated against the cost function.
            target (list[tensorops.Tensor.Tensor]): The desired output for the neural network given an input.

        Returns:
            tensorops.model.Model.loss (tensorops.Tensor.Tensor): The resulting Tensor as an output from the calculations of the neural network.
        """
        assert (
            self.model_input_layer and self.model_input_layer.layer_input_Tensors
        ), f"{type(self).__name__}.input_layer not defined!"
        assert (
            self.model_output_layer and self.model_output_layer.layer_output_Tensors
        ), f"{type(self).__name__}.output_layer not defined!"
        with self.context:
            for model_target_Tensors, training_target, model_output_Tensors, y in zip(
                self.targets, target, self.model_output_layer.layer_output_Tensors, output  # type: ignore
            ):
                model_output_Tensors.set_value(y.value)
                model_target_Tensors.set_value(training_target.value)
            self.context.recompute()
        return self.loss  # type: ignore


class FullyConnectedNetwork(SequentialModel):
    """
    A general-purpose `tensorops.model.SequentialModel` subclass used to create a ready-to-use customisable `tensorops.model.SequentialModel` fully connected network.
    """

    def __init__(
        self,
        loss_criterion: Loss,
        num_inputs: int,
        num_hidden_layers: int,
        num_hidden: int,
        num_outputs: int,
        hidden_layer_activation_function: Callable | None,
        output_layer_activation_function: Callable | None,
        seed=None,
    ) -> None:
        super().__init__(loss_criterion, seed)

        with self.context:
            self.add_layer(num_inputs, num_hidden, hidden_layer_activation_function)
            for _ in range(num_hidden_layers):
                self.add_layer(num_hidden, num_hidden, hidden_layer_activation_function)
            self.add_layer(num_hidden, num_outputs, output_layer_activation_function)
        self.loss = self.loss_criterion(
            self.targets, self.model_output_layer.layer_output_tensors
        )
