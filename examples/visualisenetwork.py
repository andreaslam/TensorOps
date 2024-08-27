

from tensorops.model import Model
from tensorops.node import Node, forward
from tensorops.tensorutils import visualise_graph
from tensorops.loss import MSELoss

class LinearModel(Model):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        with self.context:
            self.m = Node(0.7, requires_grad=True, weight=True)
            self.c = Node(0.3, requires_grad=True, weight=True)
            self.output_node = self.m * self.inputs + self.c
            self.loss = loss_criterion.loss(self.targets, self.output_node)

    def forward(self, input_node):
        with self.context:
            self.inputs.set_value(input_node.value)
            forward(self.context.nodes)
            return self.output_node

    def calculate_loss(self, output, target):
        with self.context:
            self.output_node.set_value(output.value)
            self.targets.set_value(target.value)
            return self.loss
        
if __name__ == "__main__":
    
    linear_model = LinearModel(MSELoss())
    
    visualise_graph(linear_model.context.nodes)
