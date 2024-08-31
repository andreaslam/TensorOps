# A simple `tensorops.model.Model` subclass used for demonstration purposes to create a `tensorops.model.Model` without a `tensorops.Model.Layer` class


from tensorops.model import Model
from tensorops.node import forward


class SimpleModel(Model):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        self.output_node = None

    def forward(self, model_inputs):
        assert self.output_node, "Output behaviour not defined"
        with self.context:
            self.input_nodes.set_value(model_inputs.value)
            forward(self.context.nodes)
            return self.output_node

    def calculate_loss(self, output, target):
        assert self.output_node, "Output behaviour not defined"
        with self.context:
            self.output_node.set_value(output.value)
            self.targets.set_value(target.value)
            return self.loss
