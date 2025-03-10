# This file contains code for the creation of a basic linear model y = mx + c using the tensorops.Model class
# The tensorops.visualise_graph() method visualises the computational graph created and stored in LinearModel.context (with type tensorops.NodeContext)
# The colour scheme of the nodes is as follows: salmon colour if the node is a neural network weight (m and c), pastel blue if the gradient is tracked and pastel green if the gradient is not tracked (such as the input and the output node).
# The code does not include a sample input or the full code needed for training (but includes loss function computation).


from tensorops.node import Node
from tensorops.utils.tensorutils import visualise_graph
from tensorops.loss import MSELoss
from tensorops.utils.models import SimpleSequentialModel


class LinearModel(SimpleSequentialModel):
    def __init__(self, loss_criterion):
        super().__init__(loss_criterion)
        with self.context:
            self.m = Node(0.7, requires_grad=True, weight=True)
            self.c = Node(0.3, requires_grad=True, weight=True)
            self.output_node = self.m * self.input_node + self.c
            self.loss = loss_criterion.loss(self.targets, self.output_node)


if __name__ == "__main__":
    linear_model = LinearModel(MSELoss())
    print(linear_model.context.nodes)
    visualise_graph(linear_model.context.nodes)
