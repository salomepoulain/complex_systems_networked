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
    colors = ['skyblue'] * len(network.nodesL) + ['red'] * len(network.nodesR)
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
        pos_target.update(nx.spring_layout(graph, k=0.2, iterations=50, seed=seedje))  # Update target positions

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
        node_size=50, 
        font_size=12, 
        edge_color="gray"
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
    pos = nx.spring_layout(graph, k=0.2, iterations=25, seed=seedje)
    pos_target = pos.copy()

    # Create the animation
    ani = FuncAnimation(
        fig, 
        self_sort, 
        frames=500, 
        interval=200,  
        fargs=(network, graph, colors, pos, pos_target, ax, seedje)
    )
    ani.save("animations/network_animation.gif", fps=10, writer="pillow")
    plt.show()