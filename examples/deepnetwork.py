import random
from tqdm import tqdm
from tensorops.model import Model
from tensorops.node import Node, forward, backward
from tensorops.tensorutils import PlotterUtil
from tensorops.loss import MSELoss
from tensorops.optim import SGD
from tensorops.tensorutils import visualise_graph


class DeepModel(Model):
    def __init__(self, input_size, hidden_sizes, output_size, loss_criterion):
        super().__init__(loss_criterion)
        self.layers = []
        self.biases = []
        self.activations = []
        # Initialize layers and biases
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight_layer = [
                self.add_node(
                    random.uniform(-0.5, 0.5), requires_grad=True, weight=True
                )
                for _ in range(layer_sizes[i] * layer_sizes[i + 1])
            ]
            bias_layer = [
                self.add_node(
                    random.uniform(-0.5, 0.5), requires_grad=True, weight=True
                )
                for _ in range(layer_sizes[i + 1])
            ]
            self.layers.append(weight_layer)
            self.biases.append(bias_layer)
        self.loss = Node(0.0, requires_grad=False, weight=False)

    def forward(self, input_node):
        with self.context:
            self.inputs.set_value(input_node.value)
            current_output = self.inputs

            for i, (weight_layer, bias_layer) in enumerate(
                zip(self.layers, self.biases)
            ):
                next_output = []
                for j in range(len(bias_layer)):
                    node_sum = bias_layer[j]
                    for k in range(len(weight_layer) // len(bias_layer)):
                        node_sum += (
                            current_output
                            * weight_layer[
                                j * (len(weight_layer) // len(bias_layer)) + k
                            ]
                        )
                    next_output.append(
                        node_sum.tanh()
                    )  # Using tanh activation function
                current_output = Node(
                    sum([node.value for node in next_output]), requires_grad=True
                )

            self.output_node = current_output
            forward(self.context.nodes)
            self.loss = self.loss_criterion(self.targets, self.output_node)
            return self.output_node

    def calculate_loss(self, output, target):
        with self.context:
            self.output_node.set_value(output.value)
            self.targets.set_value(target.value)
            return self.loss


if __name__ == "__main__":
    random.seed(42)

    target_m = 2.0
    target_c = 10.0

    X_train = [
        Node(random.uniform(-5, 5), requires_grad=False, weight=False)
        for _ in range(10)
    ]

    y_train = [
        Node(
            x.value * target_m + target_c + random.uniform(-1, 1),
            requires_grad=False,
            weight=False,
        )
        for x in X_train
    ]

    # Define the network architecture
    input_size = 1
    hidden_sizes = [2, 2]  # Two hidden layers with 5 nodes each
    output_size = 1

    deep_model = DeepModel(input_size, hidden_sizes, output_size, MSELoss())

    optim = SGD(deep_model.get_weights(), lr=1e-2)

    loss_plot = PlotterUtil()
    graph_plot = PlotterUtil()

    for x, y in zip(X_train, y_train):
        graph_plot.register_datapoint(
            datapoint=y.value,
            label="Training Data (TensorOps)",
            x=x.value,
            plot_style="scatter",
            colour="red",
        )

    for _ in tqdm(range(10), desc=f"Training {type(deep_model).__name__}-TensorOps"):
        for X, y in zip(X_train, y_train):
            deep_model.zero_grad()
            output = deep_model(X)
            loss = deep_model.calculate_loss(output, y)
            backward(deep_model.context.nodes)
            optim.step()
            loss_plot.register_datapoint(
                loss.value, f"{type(deep_model).__name__}-TensorOps"
            )

    print(deep_model)
    loss_plot.plot()

    for x, y in zip(X_train, y_train):
        graph_plot.register_datapoint(
            deep_model(x).value, x=x.value, label="y=mx+c (TensorOps)"
        )
    graph_plot.plot()
    # visualise_graph(deep_model.context.nodes)
