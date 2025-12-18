import matplotlib.pyplot as plt
import networkx as nx


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

    def plot(self, save_img=True, img_path="plot.png", display=True):
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
        if save_img:
            plt.savefig(img_path)
        if display:
            plt.show()


def visualise_graph(
    initial_nodes, save_img=True, img_path="graph.png", display=True
) -> None:
    """
    Visualizes an operator graph starting from a list of final (output) nodes.

    Args:
    -----
    initial_nodes (Union[list[Tensor], Tensor]): A list of Tensor/OP objects that are the final nodes of the graph to visualize. The graph is built by traversing backwards.
    save_img (bool): Whether to save the graph image to a file.
    img_path (str): Path to save the image.
    display (bool): Whether to display the graph using matplotlib.
    """
    G = nx.DiGraph()
    labels = {}

    all_nodes_map = {}

    if not initial_nodes:
        queue = []
    elif not isinstance(initial_nodes, list):
        queue = [initial_nodes]
    else:
        queue = list(initial_nodes)

    visited_ids = set()
    while queue:
        current_node = queue.pop(0)
        current_id = id(current_node)

        if current_id in visited_ids:
            continue
        visited_ids.add(current_id)
        all_nodes_map[current_id] = current_node

        if hasattr(current_node, "parents") and current_node.parents is not None:
            for parent_node in current_node.parents:
                if id(parent_node) not in all_nodes_map:
                    all_nodes_map[id(parent_node)] = parent_node
                if id(parent_node) not in visited_ids:
                    queue.append(parent_node)

    if not all_nodes_map:
        if display or save_img:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, "Empty graph", ha="center", va="center", fontsize=12)
            if save_img:
                plt.savefig(img_path, bbox_inches="tight")
            if display:
                plt.show()
            plt.close()
        return

    for node_id, node_obj in all_nodes_map.items():
        G.add_node(node_id)
        label_text = type(node_obj).__name__
        if hasattr(node_obj, "shape") and node_obj.shape is not None:
            label_text += f"\nshape={node_obj.shape}"
        labels[node_id] = label_text

    for node_id, node_obj in all_nodes_map.items():
        if hasattr(node_obj, "parents") and node_obj.parents is not None:
            for parent_node in node_obj.parents:
                parent_id = id(parent_node)
                if parent_id in all_nodes_map:
                    G.add_edge(parent_id, node_id)

    if not G.nodes:
        if display or save_img:
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, "Empty graph", ha="center", va="center", fontsize=12)
            if save_img:
                plt.savefig(img_path, bbox_inches="tight")
            if display:
                plt.show()
            plt.close()
        return

    try:
        pos = nx.planar_layout(G)
    except nx.NetworkXException:
        try:
            pos = nx.kamada_kawai_layout(G)
        except nx.NetworkXException:
            pos = nx.spring_layout(G, seed=42)

    node_colors = []
    for node_id_in_graph in G.nodes():
        node_obj = all_nodes_map[node_id_in_graph]

        color = "#C1E1C1"
        if hasattr(node_obj, "weight") and node_obj.weight:
            color = "#FFB6C1"
        elif hasattr(node_obj, "requires_grad") and node_obj.requires_grad:
            color = "#00B4D9"
        node_colors.append(color)

    fig_width = max(10, G.number_of_nodes() * 0.8 if G.number_of_nodes() > 0 else 10)
    fig_height = max(8, G.number_of_nodes() * 0.6 if G.number_of_nodes() > 0 else 8)
    plt.figure(figsize=(fig_width, fig_height))

    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_size=2500,
        node_color=node_colors,
        font_size=9,
        font_weight="normal",
        arrowsize=15,
        width=1.5,
    )

    if save_img:
        plt.savefig(img_path, bbox_inches="tight", dpi=150)
    if display:
        plt.show()
    plt.close()
