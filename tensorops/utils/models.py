from typing import Callable
from tensorops.model import Model
from tensorops.node import Node, NodeContext, forward
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
        self.context = NodeContext()
        self.loss_criterion = loss_criterion
        self.model_layers = []
        with self.context:
            self.activation_function = activation_function
            self.input_node = Node(0.0, requires_grad=False)
            if self.activation_function:
                self.output_node = self.activation_function(
                    Node(0.0, requires_grad=False)
                )
            else:
                self.output_node = Node(0.0, requires_grad=False)
            self.targets = Node(0.0, requires_grad=False)
            self.loss = loss_criterion.loss(self.targets, self.output_node)

    def forward(self, model_inputs: Node) -> Node:  # type: ignore
        assert self.output_node, "Output behaviour not defined"
        with self.context:
            self.input_node.set_value(model_inputs.value)
            forward(self.context.nodes)
            return self.output_node

    def calculate_loss(self, output: Node, target: Node) -> Node:  # type: ignore
        assert self.output_node, "Output behaviour not defined"
        with self.context:
            self.output_node.set_value(output.value)
            self.targets.set_value(target.value)
            self.input_node.trigger_recompute()
            return self.loss

    def __repr__(self) -> str:
        if [node for node in self.context.nodes if node.weight]:
            return f"{type(self).__name__}(weights={[node for node in self.context.nodes if node.weight]})"
        return "[Warning]: no weights initialised yet"


class SequentialModel(Model):
    """
    A general-purpose `tensorops.model.Model` subclass used to create a customisable `tensorops.model.Model`.

    This assumes that each layer is used sequentially without additional set-up for the forward pass.
    """

    def __init__(self, loss_criterion: Loss, seed=None) -> None:
        super().__init__(loss_criterion, seed)

    def forward(self, model_inputs: list[Node]) -> list[Node]:  # type: ignore
        assert (
            self.model_input_layer and self.model_input_layer.layer_input_nodes
        ), f"{type(self).__name__}.input_layer not defined!"
        assert (
            self.model_output_layer and self.model_output_layer.layer_output_nodes
        ), f"{type(self).__name__}.output_layer not defined!"
        assert len(model_inputs) == len(
            self.model_input_layer.layer_input_nodes
        ), f"Inputs length {len(model_inputs)} != number of input nodes of model {len(self.model_input_layer.layer_input_nodes)}"
        with self.context:
            for layer in self.model_layers:
                model_inputs = layer(model_inputs)
        return model_inputs

    def calculate_loss(self, output: list[Node], target: list[Node]) -> Node:  # type: ignore
        """
        Calculates the loss between the predicted output from the neural network against the desired output using the cost function set in `tensorops.Model.loss_criterion`.
        Args:
            output (list[tensorops.node.Node]): The prediction by the neural network to be evaluated against the cost function.
            target (list[tensorops.node.Node]): The desired output for the neural network given an input.

        Returns:
            tensorops.model.Model.loss (tensorops.node.Node): The resulting node as an output from the calculations of the neural network.
        """
        assert (
            self.model_input_layer and self.model_input_layer.layer_input_nodes
        ), f"{type(self).__name__}.input_layer not defined!"
        assert (
            self.model_output_layer and self.model_output_layer.layer_output_nodes
        ), f"{type(self).__name__}.output_layer not defined!"
        with self.context:
            for model_target_nodes, training_target, model_output_nodes, y in zip(
                self.targets, target, self.model_output_layer.layer_output_nodes, output  # type: ignore
            ):
                model_output_nodes.set_value(y.value)
                model_target_nodes.set_value(training_target.value)
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
            self.targets, self.model_output_layer.layer_output_nodes
        )
