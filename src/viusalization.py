from src.classes.network import Network
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
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
            print("iteration limit reached")
            return []
        
    self_sort.frame_counter = (self_sort.frame_counter + 1) % frames_per_alteration

    # move past zero or 20s
    # if network.give_alterations() % alterations_per_frame == 0: 
        # while network.give_alterations() % alterations_per_frame == 0: 
            # network.update_round()

    # while network.give_alterations() % alterations_per_frame != 0 and network.iterations <=limit:
        # network.update_round()

    # clear and rebuild edges in the graph
    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)

    graph.add_edge(network.removed_edge[0], network.removed_edge[1])
    # graph.add_edge(network.new_edge[0], network.new_edge[1])

    # Calculate averages for right and left nodes
    average_right, average_left = calculate_fraction(network)


    print(f"There have now been {network.give_alterations()} alterations made in {network.iterations} iterations")

    pos_target.update(nx.kamada_kawai_layout(graph, scale=0.8))
    
    # Smoothly interpolate positions between frames
    alpha = (frame % framestep) / 1000

    for node in pos:
        pos[node] = pos[node] * (1 - alpha) + pos_target[node] * alpha 

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

    # edges = nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="gray", width=0.5)

    # title = ax.set_title(f"Alterations  {network.give_alterations()}  |  Iterations  {network.iterations}", fontsize=10)

    ax.text(0.91, 0.95, f"ITERATIONS  {network.iterations}", fontsize=18, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial")

    ax.text(0.91, 0.9, f"ALTERATIONS  {network.give_alterations()}", fontsize=18, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial")

    # right average
    ax.add_patch(Circle((0.93, 0.85), 0.02, color="#FF6666", transform=ax.transAxes))
    ax.text(
        0.97, 0.85, f"{average_right:.2f}",
        fontsize=18, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial"
    )

    #left average
    ax.add_patch(Circle((0.93, 0.78), 0.02, color="lightblue", transform=ax.transAxes))
    ax.text(
        0.97, 0.78, f"{average_left:.2f}",
        fontsize=18, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial"
    )

    # return all drawn artists
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

    # fig.tight_layout(pad=5)  # leave space for title
    # Create the animation
    ani = FuncAnimation(
        fig, 
        self_sort, 
        frames=1000, 
        interval=200,  
        fargs=(network, graph, colors, pos, pos_target, ax, seedje),
        blit = True
    )
    self_sort.ani = ani
    ani.save("animations/network_animation9.gif", fps=15,  writer="ffmpeg")
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