from tensorops.model import Model
from tensorops.node import Node, NodeContext, forward
from tensorops.loss import Loss


class SimpleSequentialModel(Model):
    """
    # A simple `tensorops.model.Model` subclass used for demonstration purposes to create a `tensorops.model.Model` without a `tensorops.Model.Layer` class
    """

    def __init__(self, loss_criterion: Loss) -> None:
        super().__init__(loss_criterion)
        self.context = NodeContext()
        self.loss_criterion = loss_criterion
        self.model_layers = []
        with self.context:
            self.input_node = Node(0.0, requires_grad=False)
            self.output_node = Node(0.0, requires_grad=False)
            self.targets = Node(0.0, requires_grad=False)
            self.loss = loss_criterion.loss(self.targets, self.output_node)

    def forward(self, model_inputs: Node) -> Node:
        assert self.output_node, "Output behaviour not defined"
        with self.context:
            self.input_node.set_value(model_inputs.value)
            forward(self.context.nodes)
            return self.output_node

    def calculate_loss(self, output: Node, target: Node) -> Node:
        assert self.output_node, "Output behaviour not defined"
        with self.context:
            self.output_node.set_value(output.value)
            self.targets.set_value(target.value)
            return self.loss

    def __repr__(self) -> str:
        if [node for node in self.context.nodes if node.weight]:
            return f"{type(self).__name__}(weights={[node for node in self.context.nodes if node.weight]})"
        return "[Warning]: no weights initialised yet"


class SequentialModel(Model):
    """
    A general-purpose `tensorops.model.Model` subclass used to create a ready-to-use customisable `tensorops.model.Model`.

    This assumes that each layer is used sequentially without additional set-up for the forward pass.
    """

    def __init__(self, loss_criterion: Loss, seed=None) -> None:
        super().__init__(loss_criterion, seed)

    def forward(self, model_inputs: list[Node]) -> list[Node]:
        assert self.input_layer, f"{type(self).__name__}.input_layer not defined!"
        assert self.output_layer, f"{type(self).__name__}.output_layer not defined!"
        assert len(model_inputs) == len(
            self.input_layer.input_nodes
        ), f"Inputs length {len(model_inputs)} != number of input nodes of model {len(self.input_layer.input_nodes)}"
        with self.context:
            for layer in self.model_layers:
                model_inputs = layer(model_inputs)
        return model_inputs

    def calculate_loss(self, output: Node, target: Node) -> Node:
        """
        Calculates the loss between the predicted output from the neural network against the desired output using the cost function set in `tensorops.Model.loss_criterion`.
        Args:
            output (tensorops.node.Node): The prediction by the neural network to be evaluated against the cost function.
            target (tensorops.node.Node): The desired output for the neural network given an input.

        Returns:
            tensorops.model.Model.loss (tensorops.node.Node): The resulting node as an output from the calculations of the neural network.
        """
        with self.context:
            for model_target_nodes, training_target, model_output_nodes, y in zip(
                self.targets, target, self.output_layer.layer_output_nodes, output
            ):
                model_output_nodes.set_value(y.value)
                model_target_nodes.set_value(training_target.value)
        return self.loss  # type: ignore
