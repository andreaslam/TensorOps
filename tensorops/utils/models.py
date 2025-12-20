from typing import Callable

from tensorops.loss import Loss
from tensorops.model import Model
from tensorops.tensor import Tensor, TensorContext


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
            self.input_tensor = Tensor(0.0, requires_grad=False)
            if self.activation_function:
                self.output_tensor = self.activation_function(
                    Tensor(0.0, requires_grad=False)
                )
            else:
                self.output_tensor = Tensor(0.0, requires_grad=False)
            self.targets = Tensor(0.0, requires_grad=False)
            self.loss = loss_criterion.loss(self.targets, self.output_tensor)

    def forward(self, model_inputs: Tensor) -> Tensor:  # type: ignore
        assert self.output_tensor, "Output behaviour not defined"
        with self.context:
            for layer in self.model_layers:
                layer.forward(model_inputs)
                model_inputs = layer.layer_output
            return self.model_layers[-1].layer_output

    def calculate_loss(self, output: Tensor, target: Tensor) -> Tensor:  # type: ignore
        assert self.output_tensor and self.output_tensor.values is not None, (
            "Output layer not defined"
        )
        assert self.targets is not None, "Targets not defined"
        with self.context:
            self.output_tensor.values = output.values
            self.targets.values = target.values
            self.context.forward(recompute=True)
        return self.loss

    def __repr__(self) -> str:
        weights = [op for op in self.context.ops if hasattr(op, "weight") and op.weight]
        if weights:
            return f"{type(self).__name__}(weights={weights})"
        return "[Warning]: no weights initialised yet"


class SequentialModel(Model):
    """
    A general-purpose `tensorops.model.Model` subclass used to create a customisable `tensorops.model.Model`.

    This assumes that each layer is used sequentially without additional set-up for the forward pass.
    """

    def __init__(self, loss_criterion: Loss, seed=None, *, batch_size: int = 1) -> None:
        super().__init__(loss_criterion, seed, batch_size=batch_size)

    def forward(self, model_inputs: list[Tensor]) -> list[Tensor]:  # type: ignore
        assert self.model_input_layer and self.model_input_layer.layer_input_tensors, (
            f"{type(self).__name__}.input_layer not defined!"
        )
        assert (
            self.model_output_layer and self.model_output_layer.layer_output_tensors
        ), f"{type(self).__name__}.output_layer not defined!"
        assert len(model_inputs) == len(self.model_input_layer.layer_input_tensors), (
            f"Inputs length {len(model_inputs)} != number of input Tensors of model {len(self.model_input_layer.layer_input_tensors)}"
        )
        with self.context:
            for layer in self.model_layers:
                model_inputs = layer(model_inputs)
        return model_inputs

    def calculate_loss(self, output, target):  # type: ignore
        """
        Calculates the loss between the predicted output from the neural network against the desired output using the cost function set in `tensorops.Model.loss_criterion`.
        Args:
            output: The prediction by the neural network to be evaluated against the cost function.
            target: The desired output for the neural network given an input.

        Returns:
            tensorops.model.Model.loss (tensorops.Tensor.Tensor): The resulting Tensor as an output from the calculations of the neural network.
        """
        assert self.model_input_layer and self.model_input_layer.layer_input_tensors, (
            f"{type(self).__name__}.input_layer not defined!"
        )
        assert self.model_output_layer and self.model_output_layer.layer_output, (
            f"{type(self).__name__}.output_layer not defined!"
        )
        with self.context:
            # Handle both single Tensor and list of Tensors
            if isinstance(output, list) and isinstance(target, list):
                for (
                    model_target_tensors,
                    training_target,
                    model_output_tensors,
                    y,
                ) in zip(
                    self.targets,
                    target,
                    self.model_output_layer.layer_output_tensors,
                    output,  # type: ignore
                ):
                    model_output_tensors.values = y.values
                    model_target_tensors.values = training_target.values
                # Rebuild loss using current outputs/targets for list case
                self.loss = self.loss_criterion(
                    self.targets, self.model_output_layer.layer_output_tensors
                )
            else:
                # Single tensor case (for batched training): rebuild loss graph each call
                # This avoids stale references created during model init.
                self.loss = self.loss_criterion(output, target)
            self.context.forward(recompute=True)
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
