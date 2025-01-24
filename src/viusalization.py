from src.classes.network import Network
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np

def initial_graph(network, clusters=[]):
    """
    Create the initial graph and assign node colors.

    Args:
        network: The network object.

    Returns:
        A tuple containing the graph and the list of node colors.
    """
    index = 0
    collortjes = ["purple", "orange", "pink", "green", "yellow", "brown"]
    colors = ['skyblue'] * len(network.nodesL) + ['red'] * len(network.nodesR)
    colors = np.array(colors)
    for clus in clusters: 

        clus_array = np.array(list(clus))

        # Extract the ID of nodes within cluster
        first_items = clus_array[:, 0]

        colors[first_items] = collortjes[index]
        index+=1
        if index >= len(collortjes): 
            index = 0

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
    num_updates = 100
    for i in range(num_updates):
        network.update_round()
        all_alterations += network.alterations
    
        # not too many updates in one step
        # if all_alterations > 0.02 * len(network.all_nodes):
        #     num_updates=i
        #     break

    print(all_alterations)

    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    
    if frame % framestep == 0:
        if frame==0:
            waarde = 0
        else:
            waarde = frame - framestep
        print(f"between frames {waarde}, {frame} there were {all_alterations} alterations made ({num_updates} network updates made)")
        all_alterations= 0
        seedje += frame
        pos_target.update(nx.spring_layout(graph, k=0.5, iterations=50, seed=seedje))  # Update target positions

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
        node_size=20, 
        # font_size=12, 
        edge_color="grey", 
        width = 0.4
    )
    ax.set_title(f"frame: {frame}", fontsize=14)

def animate_network(network):
    """
    Plot and animate the network.

    Args:
        network: The network object to visualize.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    graph, colors = initial_graph(network)
    seedje = 33
    pos = nx.spring_layout(graph, k=0.5, iterations=25, seed=seedje)
    pos_target = pos.copy()

    # Create the animation
    ani = FuncAnimation(
        fig, 
        self_sort, 
        frames=200, 
        interval=100,  
        fargs=(network, graph, colors, pos, pos_target, ax, seedje)
    )
    ani.save("animations/network_animation.gif", fps=20, writer="pillow")
    plt.show()

def plot_network(network, cluster):
    fig, ax = plt.subplots(figsize=(6, 6))
    graph, colors = initial_graph(network, cluster)
    seedje = 33
    # pos = nx.spring_layout(graph, k=2, iterations=50, seed=seedje)

    density = nx.density(graph)
    k = 0.1 / (1 + density)  # Smaller k for denser graphs
    pos = nx.spring_layout(graph, k=0.01, iterations=200, seed=seedje)
    # pos = nx.circular_layout(graph)
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    nx.draw(
        graph, 
        pos,
        ax=ax,
        with_labels=False, 
        node_color=colors, 
        node_size=100, 
        # font_size=12, 
        edge_color="grey", 
        width = 0.4
    )
    ax.set_title(f"animating cluster connectivity", fontsize=14)

    plt.show()


def plot_cascade_dist(data):
    # Prepare the figure
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Plot each cascade size with vertical stacking
    for size, values in data.items():
        if size==1:
            continue
        y_offsets = [y * 0.5 for y in range(len(values))]
        for (index, polarization), y in zip(values, y_offsets):
            color = plt.cm.coolwarm((polarization + 1) / 2)  # Map polarization (-1 to 1) to colormap
            ax.scatter(size, y, color=color, s=30)  # Plot the dot
            # ax.annotate(str(index), (size, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    # Add labels and title
    ax.set_title("Cascade Sizes and Polarizations")
    ax.set_xlabel("Cascade Size")
    ax.set_ylabel("Stacked Occurrences")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlim(1)
    ax.set_ylim(1)
    ax.grid(True)

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Polarization (-1: Red, 1: Blue)")

    # Show the plot
    plt.tight_layout()
    plt.show()