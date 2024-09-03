# A simple `tensorops.model.Model` subclass used for demonstration purposes to create a `tensorops.model.Model` without a `tensorops.Model.Layer` class


from tensorops.model import Model
from tensorops.node import Node, NodeContext, forward


class SimpleSequentialModel(Model):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        self.context = NodeContext()
        self.loss_criterion = loss_criterion
        self.model_layers = []
        with self.context:
            self.input_node = Node(0.0, requires_grad=False)
            self.output_node = Node(0.0, requires_grad=False)
            self.targets = Node(0.0, requires_grad=False)
            self.loss = loss_criterion.loss(self.targets, self.output_node)

    def forward(self, model_inputs):
        assert self.output_node, "Output behaviour not defined"
        with self.context:
            self.input_node.set_value(model_inputs.value)
            forward(self.context.nodes)
            return self.output_node

    def calculate_loss(self, output, target):
        assert self.output_node, "Output behaviour not defined"
        with self.context:
            self.output_node.set_value(output.value)
            self.targets.set_value(target.value)
            return self.loss

    def __repr__(self):
        if [node for node in self.context.nodes if node.weight]:
            return f"{type(self).__name__}(weights={[node for node in self.context.nodes if node.weight]})"
        return "[Warning]: no weights initialised yet"
