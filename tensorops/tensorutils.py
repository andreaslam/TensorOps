import matplotlib.pyplot as plt
import networkx as nx
import random

plt.style.use("seaborn")


class PlotterUtil:

    """
    `tensorops.PlotterUtil` is a utility class for plotting graphical data.

    Attributes
    ----------
    x_label (string): The label for the x-axis.
    y_label (string): The label for the y-axis.
    datapoints (dict[string, Union[int, float]]): Contains all data within a single category of the plot.
    labels (list[string]): A list containing labels of each category of data being plotted.
    xs (dict[string, Union[int, float]]): Contains the x-axis data for each category of data.
    plot_styles ({dict[string, string]}): The type of plot for each type of category of data
    colours (list[string]): List of colours for each category of data.
    """

    def __init__(self, x_label="Epoch", y_label="Loss"):
        self.x_label = x_label
        self.y_label = y_label
        self.datapoints = {}
        self.labels = []
        self.xs = {}
        self.plot_styles = {}
        self.colours = {}

    def register_datapoint(
        self, datapoint, label, x=None, plot_style="line", colour=None
    ):
        """
        Registers a set of coordinates to be stored inside a `tensorops.PlotterUtil` instance.

        Args:
            datapoint (Union[int, float]): the y coordinate of the new data entry to be plotted
            label (string): the legend for the graph
            x=None (Optional[Union[int, float]]): the x coordinate of the new data entry. By default the x-axis would be determined by the index of the entry, incrementing by one.
            plot_style="line" (string): the type of plot the new data is presented in.
            colour=None (Optional[string]): the colour of the new data point. By default there is no colour option listed, allowing `matplotlib` to pick a default.
        """

        if label not in self.labels:
            self.labels.append(label)
            self.datapoints[label] = []
            self.xs[label] = []
            self.plot_styles[label] = plot_style
            self.colours[label] = colour

        self.datapoints[label].append(datapoint)

        if x is not None:
            self.xs[label].append(x)
        else:
            self.xs[label].append(len(self.datapoints[label]) - 1)

    def plot(self):
        """
        Utility function that plots the stored data within the tensorops.PlotterUtil class.
        """
        for label in self.labels:
            colour = self.colours.get(label)

            if self.plot_styles[label] == "line":
                plt.plot(
                    self.xs[label], self.datapoints[label], label=label, color=colour
                )
            elif self.plot_styles[label] == "scatter":
                plt.scatter(
                    self.xs[label], self.datapoints[label], label=label, color=colour
                )

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend()
        plt.show()
        plt.close()


def visualise_graph(nodes):
    """
    Utility function that visualises the relationship between nodes as a computational graph.

    The colour scheme is as follows:
    - Salmon colour if the node is a neural network weight. (`Node.weight=True`)
    - Pastel blue if the node requires gradient. (`Node.requires_grad = True`)
    - Pastel green if it does not requires gradient. (`Node.requires_grad = False`)

    Args:
        nodes (list[tensorops.Node]): The collection nodes to be examined.
    """
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
