from src.classes.network import Network
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def initial_graph(network, clusters=[]):
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

from matplotlib.patches import Rectangle

def self_sort(frame, network, graph, colors, pos, pos_target, ax, seedje):
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
    print(f"Processing frame {frame}")
    framestep = 200
    limit = 50000
    alterations_per_frame = 1
    frames_per_alteration = 15
    ax.clear()

    # static counter to track sub-frames
    if not hasattr(self_sort, "frame_counter"):
        self_sort.frame_counter = 0

    if self_sort.frame_counter == 0:
        current_alteration = network.give_alterations()

        while network.give_alterations() == current_alteration and network.iterations <= limit:
            network.update_round()

        # stop animation if desired iteration limit is reached
        if network.iterations >= limit:
            self_sort.ani.event_source.stop()
            print("Iteration limit reached")
            return []

    self_sort.frame_counter = (self_sort.frame_counter + 1) % frames_per_alteration

    # clear and rebuild edges in the graph
    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)

    # re-add the removed edge for visualization
    graph.add_edge(network.removed_edge[0], network.removed_edge[1])

    # calculate averages for right and left nodes
    average_right, average_left = calculate_fraction(network)

    # define kawai layout <:-)
    pos_target.update(nx.kamada_kawai_layout(graph, scale=0.8))

    # smoothly interpolate positions between frames
    alpha = (frame % framestep) / 1000

    # 
    for node in pos:
        pos[node] = pos[node] * (1 - alpha) + pos_target[node] * alpha

    # draw nodes
    nodes = nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=400)

    # draw regular edges in gray
    regular_edges = [
        edge for edge in graph.edges if edge != network.removed_edge and edge != network.new_edge
    ]
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=regular_edges, edge_color="gray", width=0.5)

    # draw removed edge in red
    if network.removed_edge in graph.edges:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[network.removed_edge], edge_color="red", width=1.5)

    # draw new edge in green
    if network.new_edge in graph.edges:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[network.new_edge], edge_color="green", width=1.5)

    # add title
    ax.text(
    0.98, 0.99, 
    f"ITERATIONS  {network.iterations}", 
    fontsize=16, 
    color="gray", 
    transform=ax.transAxes, 
    ha="left", 
    va="center", 
    fontfamily="Arial"  
    )

    ax.text(
    0.98, 0.96, 
    f"ALTERATIONS  {network.give_alterations()}", 
    fontsize=16, 
    color="gray", 
    transform=ax.transAxes, 
    ha="left", 
    va="center", 
    fontfamily="Arial" 
    )

    # red circle for right average
    ax.add_patch(Circle((1, 0.93), 0.02, color="red", transform=ax.transAxes))
    text_right = ax.text(
        0.98, 0.8, f"Right Avg: {average_right:.2f}",
        fontsize=16, color="black", transform=ax.transAxes, ha="left", va="center"
    )

    # blue circle for left average
    ax.add_patch(Circle((1, 0.9), 0.02, color="blue", transform=ax.transAxes))
    text_left = ax.text(
        0.98, 0.7, f"Left Avg: {average_left:.2f}",
        fontsize=16, color="black", transform=ax.transAxes, ha="left", va="center"
    )

    # Return all drawn artists
    return [nodes]

def plot_network(network):
    """
    Plot and animate the network.

    Args:
        network: The network object to visualize.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    graph, colors = initial_graph(network)

    for _ in range(len(network.all_nodes) // 2):
        node_a, node_b = np.random.choice(len(network.all_nodes), 2, replace=False)
        graph.add_edge(node_a, node_b)
    
    # pos = nx.spring_layout(graph, k=0.2, iterations=25, seed=seedje)
    seedje = 33
    pos = nx.kamada_kawai_layout(graph, scale=0.6)
    pos_target = pos.copy()

    for spine in ax.spines.values():  # remove borders
        spine.set_visible(False)

    # leave space for title
    # Create the animation
    ani = FuncAnimation(
        fig, 
        self_sort, 
        frames=200, 
        interval=200,  
        fargs=(network, graph, colors, pos, pos_target, ax, seedje),
        blit = True
    )
    self_sort.ani = ani
    ani.save("animations/network_animation10.gif", fps=15,  writer="ffmpeg")
    plt.show()

def plot_network_clusters(network, cluster):
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

def plot_cascade_dist_average(data, stadium, largest_size=120, num_exp=1, save=False, correlation=0):
    """
    Plot a distribution of cascade sizes with bar heights representing the number of occurrences
    and bar colors representing the average polarization.
    
    Args:
        data: A dictionary where keys are cascade sizes and values are lists of (index, polarization) tuples.
    """
    # Prepare data for the plot
    sizes = []
    counts = []
    avg_polarizations = []

    # Create a custom green-to-red colormap
    green_to_red = plt.cm.RdYlGn_r

    for size, values in data.items():
        # if size == 1:  # Skip size 1 if necessary
        #     continue
        sizes.append(size)  # The cascade size
        counts.append(len(values))  # Number of occurrences (bar height)
        avg_polarizations.append(np.mean(np.abs(values)))
    counts=np.array(counts, dtype=np.float64)
    counts/=num_exp
    
    # Normalize polarization for coloring
    colors = [green_to_red(p) for p in avg_polarizations]

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(sizes, counts, color=colors, edgecolor="black", linewidth=0.5)

    # Add a colorbar for polarization
    sm = plt.cm.ScalarMappable(cmap=green_to_red, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("0:non polarized-1:fully polarized")
    
    ax.set_yscale("log")
    # Add labels and title
    ax.set_title(f"Cascade Size Distribution with Polarization ({num_exp} runs)")
    ax.set_xlabel("Cascade Size")
    ax.set_ylabel("Number of Occurrences")
    ax.set_xlim(0, largest_size+1) 
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

    
    plt.tight_layout()

    # save the plot
    if save:
        plt.savefig(f"plots/cascade_distribution_{stadium}_{correlation}.png", dpi=300, bbox_inches='tight')

    plt.show()

def print_network(network):
    """
    Print network at one single iteration

    Args:
        network: The network object to visualize.
    """

    color_map = ['lightblue'] * len(network.nodesL) + ['#FF6666'] * len(network.nodesR)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(network.all_nodes)))
    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    


    # Set positions and draw the graph
    plt.figure(figsize=(16,8))
    pos = nx.kamada_kawai_layout(graph, scale=0.6)
    nx.draw(
        graph,
        pos,
        node_color=color_map,
        with_labels=True,
        edge_color="lightgray",
        width=0.2,
        node_size=400,
        font_size=10,
    )
    plt.show

def calculate_fraction(network):
    fraction_right = 0
    fraction_left = 0

    for i in network.all_nodes:
        identity = i.identity
        right = 0
        left = 0
        for j in i.node_connections:
            if j.identity == "R":
                right += 1

            if j.identity == "L":
                left += 1

        connections = len(i.node_connections)

        if identity == "R" and connections != 0:
            fraction_right += right / connections
        
        if identity == "L" and connections != 0:
            fraction_left += left / connections
        
    average_right = fraction_right / (len(network.all_nodes) / 2)
    average_left = fraction_left / (len(network.all_nodes) / 2)

    return average_right, average_left



###############################################################################################################
###############################################################################################################
###############################################################################################################

        # num_updates = 1000
    # for i in range(num_updates):
        # network.update_round()
        # all_alterations += network.alterations

        # not too many updates in one step
        # if all_alterations > 0.02 * len(network.all_nodes):
            # num_updates=i
            # break

    # if frame % framestep == 0:
    #     if frame==0:
    #         waarde = 0
    #     else:
    #         waarde = frame - framestep
    #     # print(f"between frames {waarde}, {frame} there were {all_alterations} alterations made ({num_updates} network updates made)")
        
    #     all_alterations= 0
    #     seedje += frame
    #     # pos_target.update(nx.spring_layout(graph, k=0.2, iterations=50, seed=seedje))


    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################

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