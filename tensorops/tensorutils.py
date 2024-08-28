import matplotlib.pyplot as plt
import networkx as nx


class LossPlotter:
    def __init__(self):
        self.datapoints = {}
        self.labels = []
        self.xs = {}

    def register_datapoint(self, datapoint, label, x=None):
        if label not in self.labels:
            self.labels.append(label)
            self.datapoints[label] = []
            self.xs[label] = []

        self.datapoints[label].append(datapoint)

        if x is not None:
            self.xs[label].append(x)
        else:
            self.xs[label].append(len(self.datapoints[label]) - 1)

    def plot(self):
        for label in self.labels:
            plt.plot(self.xs[label], self.datapoints[label], label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.close()


def visualise_graph(nodes):
    G = nx.DiGraph()
    labels = {}
    for node in nodes:
        node_id = id(node)
        if node.value and node.grad:
            node_label = f"{type(node).__name__}\nVal: {round(node.value, 2)}\nGrad: {round(node.grad, 2)}"
        else:
            node_label = f"{type(node).__name__}\nVal: {node.value}\nGrad: {node.grad}"
        labels[node_id] = node_label
        G.add_node(node_id)
        for parent in node.parents:
            parent_id = id(parent)
            G.add_edge(parent_id, node_id)

    pos = nx.planar_layout(G)
    colourmap = [
        "#FFB6C1" if node.weight else "#00B4D9" if node.requires_grad else "#C1E1C1"
        for (_, node) in zip(G, nodes)
    ]
    # salmon colour if the node is a neural network weight, pastel blue and green if the node requires grad and if it does not respectively.

    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=800,
        node_color=colourmap,
        font_size=6,
    )
    plt.show()
