from src.classes.network import Network
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np

def initial_graph(network):
    """
    Create the initial graph and assign node colors.

    Args:
        network: The network object.

    Returns:
        A tuple containing the graph and the list of node colors.
    """
    colors = ['lightblue'] * len(network.nodesL) + ['#FF6666'] * len(network.nodesR)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(network.all_nodes)))
    return graph, colors

def self_sort(frame, network, graph, colors, pos, pos_target, ax, seedje, all_alterations=0):
    """
    Update the graph for each animation frame.

    Args:
        frame: Current frame number.
        network: The network object.
        graph: The NetworkX graph object.
        colors: List of node colors.
        pos: Current node positions.
        pos_target: Target positions for interpolation.
        ax: Matplotlib axis object.
        seedje: Seed for random layout generation.
    """
    framestep = 50
    ax.clear()

    for _ in range(100):
        network.update_round()
        all_alterations += network.alterations

    
    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    
    if frame % framestep == 0:
        print(f"between frames {frame}, {frame+framestep} there were {all_alterations} alterations made")
        all_alterations= 0
        seedje += frame
        # pos_target.update(nx.spring_layout(graph, k=0.2, iterations=50, seed=seedje))
        pos_target.update(nx.kamada_kawai_layout(graph, scale=0.8))
    # Smoothly interpolate positions between frames
    alpha = (frame % framestep) / 1000
    for node in pos:
        pos[node] = pos[node] * (1 - alpha) + pos_target[node] * alpha 
    
    nx.draw(
        graph, 
        pos,
        ax=ax,
        with_labels=False, 
        node_color=colors, 
        # node_size=50, 
        # font_size=12, 
        edge_color="lightgray",  
        width=0.2,
        node_size=80,
        font_size=10,
    )
    ax.set_title(f"frame: {frame}", fontsize=14)

def plot_network(network):
    """
    Plot and animate the network.

    Args:
        network: The network object to visualize.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    graph, colors = initial_graph(network)
    seedje = 33
    # pos = nx.spring_layout(graph, k=0.2, iterations=25, seed=seedje)
    pos = nx.kamada_kawai_layout(graph, scale=0.8)
    pos_target = pos.copy()

    # Create the animation
    ani = FuncAnimation(
        fig, 
        self_sort, 
        frames=1000, 
        interval=200,  
        fargs=(network, graph, colors, pos, pos_target, ax, seedje)
    )
    ani.save("animations/network_animation.gif", fps=10, writer="pillow")
    plt.show()


def print_network(network):
    """
    Print network.
    """

    color_map = ['lightblue'] * len(network.nodesL) + ['#FF6666'] * len(network.nodesR)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(network.all_nodes)))
    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    


    # Set positions and draw the graph
    plt.figure(figsize=(8,8))
    pos = nx.kamada_kawai_layout(graph, scale=0.8)
    nx.draw(
        graph,
        pos,
        node_color=color_map,
        with_labels=True,
        edge_color="lightgray",
        width=0.2,
        node_size=500,
        font_size=10,
    )
    plt.show

    # def plot_degree_distribution(self):
    #     # calculate degrees of all nodes
    #     degrees = [deg for _, deg in self.graph.degree()]

    #     # count frequencies of each degree
    #     degree_counts = Counter(degrees)

    #     # sort by degree
    #     degrees, counts = zip(*sorted(degree_counts.items()))

    #     # plot the degree distribution
    #     plt.figure(figsize=(8, 6))
    #     plt.bar(degrees, counts, width=0.8, color="blue", edgecolor="black", alpha=0.7)
    #     plt.title("Degree Distribution")
    #     plt.xlabel("Degree")
    #     plt.ylabel("Frequency")
    #     plt.grid(True, linestyle="--", alpha=0.7)
    #     plt.show()