import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from collections import defaultdict
from scipy.stats import ttest_ind_from_stats
import networkx as nx
import numpy as np
import powerlaw
import matplotlib.colors as mcolors
from PIL import Image, ImageSequence
import os


def initial_graph(network):
    """
    Create an initial graph and assign node colors.

    Args:
        network: The network object.

    Returns:
        A tuple containing the graph and the list of node colors.
    """
    colors = ['lightblue'] * len(network.nodesL) + ['#FF6666'] * len(network.nodesR)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(network.all_nodes)))

    return graph, colors


def self_sort(frame, network, graph, colors, pos, pos_target, ax):
    """
    Update graph for each animation frame.

    Args:
        frame: Current frame number.
        network: The network object.
        graph: The NetworkX graph object.
        colors: List of node colors.
        pos: Current node positions.
        pos_target: Target positions for interpolation.
        ax: Matplotlib axis object.
        seedje: Seed for random layout generation.

    Returns: 
        Nodes in the graph
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
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=regular_edges, edge_color="lightgray", width=0.5)
    # nx.draw(graph, pos, node_color=colors, ax=ax, with_labels=False, edge_color="lightgray", width=0.5)
    
    # draw removed edge in red
    if network.removed_edge in graph.edges:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[network.removed_edge], edge_color="red", width=1.5)
        # nx.draw(graph, pos, node_color=colors, ax=ax, with_labels=False, edge_color="lightgray", width=0.5)
    # draw new edge in green
    if network.new_edge in graph.edges:
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=[network.new_edge], edge_color="green", width=1.5)

    # add title
    ax.text(
    0.94, 1, 
    f"ITERATIONS   {network.iterations}", 
    fontsize=16, 
    color="gray", 
    transform=ax.transAxes, 
    ha="left", 
    va="center", 
    fontfamily="Arial"  
    )

    ax.text(
    0.94, 0.96, 
    f"ALTERATIONS  {network.give_alterations()}", 
    fontsize=16, 
    color="gray", 
    transform=ax.transAxes, 
    ha="left", 
    va="center", 
    fontfamily="Arial" 
    )

    # red circle for right average
    ax.add_patch(Circle((0.96, 0.91), 0.02, color="#FF6666", transform=ax.transAxes))
    text_right = ax.text(
        0.99, 0.91, f"           {average_right:.2f}",
        fontsize=16, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial"
    )

    # blue circle for left average
    ax.add_patch(Circle((0.96, 0.85), 0.02, color="lightblue", transform=ax.transAxes))
    text_left = ax.text(
        0.99, 0.85, f"           {average_left:.2f}",
        fontsize=16, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial"
    )

    # Return all drawn artists
    return [nodes]

def plot_network(network, i):
    """
    Animate the network.

    Args:
        network: The network object to visualize. 
        i: Integer to make the filename unique
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    graph, colors = initial_graph(network)

    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
        
    # pos = nx.spring_layout(graph, k=0.2, iterations=25, seed=seedje)
    seedje = 33
    pos = nx.kamada_kawai_layout(graph, scale=0.6)
    pos_target = pos.copy()

    # remove borders
    for spine in ax.spines.values():  
        spine.set_visible(False)

    # leave space for title
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
    ani.save(f"animations/network_animation{i}.gif", fps=15,  writer="ffmpeg")
    plt.show()

def plot_network_clusters(network, cluster):
    """
    Plots a network with clusters using a force-directed layout.

    Parameters:
    network: The network object to visualize. 
    cluster: Cluster-related data for node coloring and graph structure.
    """
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
        node_size=400, 
        # font_size=12, 
        edge_color="grey", 
        width = 0.4
    )
    ax.set_title(f"animating cluster connectivity", fontsize=14)

    plt.show()


def create_distribution(data, num_exp=1):
    """
    This function iterates through a list of cascades and extracts the size of the cascade, 
    the number of times this cascade size occurred and
    the average polarization of this cascade size. 

    args:
        data: dictionary of the sizes with their cascade list of political orientations of nodes within that cascade.
        num_exp: number of runs done.
    returns:
        counts: list of frequencies a particular size occured (matched index-wise with size)
        sizes: list of sizes that occurred
        avg_polarizations: average polarizaiton of that particular cascade size (matched index-wise with size)
    """
def plot_cascade_dist(data):
    """
    Plots cascade sizes and their polarization levels using a stacked scatter plot.

    Parameters:
    data: A dictionary where keys are cascade sizes and values are lists of (index, polarization) tuples.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # plot each cascade size with vertical stacking
    for size, values in data.items():
        if size==1:
            continue
        y_offsets = [y * 0.5 for y in range(len(values))]
        for (index, polarization), y in zip(values, y_offsets):
            color = plt.cm.coolwarm((polarization + 1) / 2)  
            ax.scatter(size, y, color=color, s=30) 
            # ax.annotate(str(index), (size, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    # add labels and title
    ax.set_title("Cascade Sizes and Polarizations")
    ax.set_xlabel("Cascade Size")
    ax.set_ylabel("Stacked Occurrences")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlim(1)
    ax.set_ylim(1)
    ax.grid(True)

    # add a colorbar
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Polarization (-1: Red, 1: Blue)")

    # show the plot
    plt.tight_layout()
    plt.show()

def plot_cascade_dist_average(data, stadium, largest_size=120, num_exp=1, save=False, correlation=0):
    """
    Plot a distribution of cascade sizes with bar heights representing the number of occurrences
    and bar colors representing the average polarization.
    
    Args:
        data: A dictionary where keys are cascade sizes and values are lists of (index, polarization) tuples.
    """
    # prepare data for the plot
    sizes = []
    counts = []
    avg_polarizations = []

    # values essentially holds the list of the orientation of nodes (+1 for left, -1 for right) within a cascade
    for size, values in data.items():
        sizes.append(size)  # The cascade size
        counts.append(len(values))  # Number of occurrences 

        # the polarizations are already weighted by its prevelences
        polarization_val = np.mean(np.abs(values))
        
        if np.isnan(polarization_val):
            print(f"Warning: NaN detected in mean absolute values for input {values}")
            polarization_val = 0  # Default to 0 or another fallback value
        

        avg_polarizations.append(polarization_val)
    counts=np.array(counts, dtype=np.float64)

    #normalize by dividng through number of experiments
    counts/=num_exp
    
    # Normalize polarization for coloring
    colors = [green_to_red(p) for p in avg_polarizations]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(sizes, counts, color=colors, edgecolor="black", linewidth=0.5, s=20)
    # create the bar plot
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(sizes, counts, color=colors, edgecolor="black", linewidth=0.5)

    # add a colorbar for polarization
    sm = plt.cm.ScalarMappable(cmap=green_to_red, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("0:non polarized-1:fully polarized")
    
    # set scales to log log for powerlaw detectio
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Add labels and title
    ax.set_title(f"Cascade Size Distribution with Polarization (correlation: {correlation})")
    ax.set_xlabel("Cascade Size")
    ax.set_ylabel("Number of Occurrences")
    ax.set_xlim(0, largest_size+1) 
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # save the plot
    if save:
        if not averaged: 
            plt.savefig(f"plots/experiment_results/cascade_distribution/{what_net}/{stadium}_{correlation}.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"plots/experiment_results/cascade_distribution/{what_net}/averaged_{stadium}_{correlation}.png", dpi=300, bbox_inches='tight')

    plt.show()


def calculate_average_per_gamma(cas, num_runs): 
    cascades_before_averaged, cascades_after_averaged = cas
    values_bef, values_af = defaultdict(), defaultdict()
    variance_bef, variance_af = defaultdict(), defaultdict()
    # variance_size_bef, variance_size_af = defaultdict(), defaultdict()

    for key,value in cascades_before_averaged.items():
        counts_bef, sizes_bef, pol_bef = create_distribution(value, num_runs)

        
        # weight sizes by their prevelence
        mean_size_bef = np.average(sizes_bef, weights=counts_bef)

        # the polarization is already weighted by prevelence so is not weighted again
        mean_pol_bef = np.mean(pol_bef) 
        values_bef[key] = (mean_size_bef, mean_pol_bef)    
        size_var_bef = np.average((sizes_bef-mean_size_bef)**2, weights=counts_bef)
        pol_var_bef = np.average((pol_bef-mean_pol_bef)**2, weights=counts_bef)
        variance_bef[key] = (size_var_bef, pol_var_bef)

    for key,value in cascades_after_averaged.items():
        counts_af, sizes_af, pol_af = create_distribution(value, num_runs)
        mean_size_af = np.average(sizes_af, weights=counts_af)
        mean_pol_af = np.mean(pol_af) 
        values_af[key] = (mean_size_af, mean_pol_af)    
        size_var_af = np.average((sizes_af-mean_size_af)**2, weights=counts_af)
        pol_var_af = np.average((pol_af-mean_pol_af)**2, weights=counts_af)
        variance_af[key] = (size_var_af, pol_var_af)

    return values_bef, values_af, variance_bef, variance_af


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
    


    # set positions and draw the graph
    plt.figure(figsize=(8,8))
    pos = nx.kamada_kawai_layout(graph, scale=0.6)
    nx.draw(
        graph,
        pos,
        node_color=color_map,
        with_labels=False,
        edge_color="lightgray",
        width=0.2,
        node_size=400,
        font_size=10,
    )
    plt.show

    average_right, avereage_left = calculate_fraction(network)

    return average_right, avereage_left

def calculate_fraction(network):
    """
    Calculate the fraction of neighbors that share the same identity

    Args:
        network: The network object.
    """
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

def update(frame, graph, pos, activated, active_color_map, color_palette, node_colors, ax, removed_edge, new_edge, inactive_samplers, network):
    """
    Update function for each frame of the animation showing a cascade

    Args:
    frame: the frame number
    graph: the current graph of the network
    pos: the current position of nodes
    activated: a list containing activated nodes
    active_color_map: the colors that active nodes should take
    color_palette: the current colors of the nodes
    node_colors: the actual colors of the nodes
    ax: Matplotlib axis object.
    removed_edge: the edge that is removed in a iteration (if applicable)
    new_edge: the new edge that is created in a iteration (if applicable)
    inactive_samplers: samplers that did not get active
    network: the network object
    """
    ax.clear()
    
    if frame < len(activated):
        if frame == 0:
            # frame 0: all nodes remain their initial color
            pass
        elif frame == 1:
            # frame 1: first set of activated nodes turn orange
            for node_id in activated[1]:
                node_colors[node_id] = "orange"
        elif frame == 2:
            # frame 2: samplers who activate turn green
            for node_id in activated[2]:
                node_colors[node_id] = color_palette[0]  
        elif frame == 3:
            # frame 3: inactive samplers revert to old colors
            half_size = len(graph.nodes) // 2  

            for node_id in inactive_samplers:
                # revert to original color
                node_colors[node_id] = "lightblue" if list(graph.nodes).index(node_id) < half_size else "#FF6666"
        else:
            # subsequent frames: new nodes in activated turn progressively different greens
            color_idx = (frame - 4) % len(color_palette) + 1
            for node_id in activated[frame - 1]:
                if node_id not in active_color_map:
                    active_color_map[node_id] = color_palette[color_idx]
                node_colors[node_id] = active_color_map[node_id]

    # draw the graph
    if frame == len(activated) and removed_edge:
        nx.draw(graph, pos, node_color=node_colors, ax=ax, with_labels=False, edge_color="lightgray", width=0.5)
        nx.draw_networkx_edges(graph, pos, edgelist=[removed_edge], edge_color="red", width=1.5)
    elif frame == len(activated) + 1 and new_edge:
        graph.remove_edge(*removed_edge)
        nx.draw(graph, pos, node_color=node_colors, ax=ax, with_labels=False, edge_color="lightgray",width=0.5)
        nx.draw_networkx_edges(graph, pos, edgelist=[new_edge], edge_color="green", width=1.5)
    elif frame == len(activated) + 2 and removed_edge:
        graph.add_edge(*new_edge)
        nx.draw(graph, pos, node_color=node_colors, ax=ax, with_labels=False, edge_color="lightgray", width=0.5)
    else:
        nx.draw(graph, pos, node_color=node_colors, ax=ax, with_labels=False, edge_color="lightgray", width=0.5)

    for node in network.activated:
        node.reset_activation_state()

    for node in network.all_nodes:
        node.sampler_state = False

    # calculate averages for right and left nodes
    average_right, average_left = calculate_fraction(network)

    # add title
    ax.text(
    0.94, 1, 
    f"ITERATIONS   {network.iterations}", 
    fontsize=16, 
    color="gray", 
    transform=ax.transAxes, 
    ha="left", 
    va="center", 
    fontfamily="Arial"  
    )

    ax.text(
    0.94, 0.96, 
    f"ALTERATIONS  {network.give_alterations()}", 
    fontsize=16, 
    color="gray", 
    transform=ax.transAxes, 
    ha="left", 
    va="center", 
    fontfamily="Arial" 
    )

    # red circle for right average
    ax.add_patch(Circle((0.96, 0.91), 0.02, color="#FF6666", transform=ax.transAxes))
    text_right = ax.text(
        0.99, 0.91, f"           {average_right:.2f}",
        fontsize=16, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial"
    )

    # blue circle for left average
    ax.add_patch(Circle((0.96, 0.85), 0.02, color="lightblue", transform=ax.transAxes))
    text_left = ax.text(
        0.99, 0.85, f"           {average_left:.2f}",
        fontsize=16, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial"
    )
            
    network.activated = set()

def simulate_cascade(network):
    """
    This function can be called to simulate one single cascade

    Args:
        network: the network object

    returns: 
        sL: s for left news sources
        sR: s for right news sources
        activated: active nodes
        inactive_samplers: inactive nodes after being sampled
    """
    network.iterations += 1 
    sL, sR = network.generate_news_significance()
    allsamplers = network.pick_samplers()
    samplers_id = []

    for sampler in allsamplers:
        for node in sampler:
            node.make_sampler()
            samplers_id.append(node.ID)
            network.activated.add(node)

    activated = network.run_cascade_for_visuals(sL, sR)
    active_samplers = list(set(activated[0]).intersection(samplers_id))
    first_round = list(set(activated[0]).difference(active_samplers))
    inactive_samplers = list(set(samplers_id) - set(active_samplers))
    activated.pop(0)
    activated.insert(0, [])
    activated.insert(1, samplers_id)
    activated.insert(2, active_samplers)
    if first_round:
        activated.insert(3, first_round)

    return sL, sR, activated, inactive_samplers

def network_adjustment_after_cascade(network, sL, sR):
    """
    This function can be called to simulated a network adjustment after having simulated a cascade

    argset:
        sL: s for left news sources
        sR: s for right news sources
        network: the network object/

    returns: 
        the removed_edge and the new_edge

    """ 
    network.network_adjustment(sL,sR)
    removed_edge = tuple(network.removed_edge)
    new_edge = tuple(network.new_edge)

    return removed_edge, new_edge


def plot_cascade(network, animation_number):
    """
    Visualize a cascade through an animation where nodes become activated over time.

    Args:
        network: the network object
        animation_number: the amount of cascades to visualize, not this creates seperate gif files

    Returns:
        old_network_connections: connections as they were before the cascade
        network.connections: connections as they are after the cascade (can be the same)
        len(network.nodesL): the amount of L nodes
        len(network.nodesR): the amount of R nodes
        change: bool that is True if there was a change in the network
        average_right: the fraction of neighbors that share the same identity and the average thereof for right
        average_left: the fraction of neighbors that share the same identity and the average thereof for left
        iterations: the iteration number
        alterations: the amount of alterations
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    old_network_connections = network.connections.copy()
    sL, sR, activated, inactive_samplers = simulate_cascade(network)

    print(activated)

    # initialize graph and colors
    graph, node_colors = initial_graph(network)
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)

    # obtain new edge and removed edge
    removed_edge, new_edge = network_adjustment_after_cascade(network, sL, sR)
    print(new_edge)
    if new_edge:
        change = True
    else:
        change = False

    # color map for activations in different "rounds" 
    color_palette = ["#008000", "#33CC33", "#66FF66", "#99FF99", "#99FF99", "#99FF99", "#99FF99"]
    # to keep track of which node is active and has what color
    active_color_map = {}

    # set up system with Kawaii
    pos = nx.kamada_kawai_layout(graph, scale=0.6)

    # create animation
    anim = FuncAnimation(
        fig,
        update,
        fargs=(graph, pos, activated, active_color_map, color_palette, node_colors, ax, removed_edge, new_edge, inactive_samplers, network),
        frames = len(activated) + 3 if change else len(activated),
        interval=500,
        repeat=False,
    )

    # save and close
    anim.save(f"animations/cascade{animation_number}.gif", writer="pillow", fps=1)
    plt.close(fig)

    average_right, average_left = calculate_fraction(network)
    iterations = network.iterations
    alterations = network.alterations

    return old_network_connections, network.connections, len(network.nodesL), len(network.nodesR), change, average_right, average_left, iterations, alterations

def update_for_gif(frame, graph, network1, network2, pos, pos_target, ax, colors, average_right, average_left, iterations, alterations):
    """
    update function for the merging gif animation

    Args:
    frame: the frame number
    graph: the current graph of the network
    network1: the network before an alteration
    network2: the network aftera an alteration
    pos: the position of nodes in network1
    pos_target: the position of nodes in the new network
    ax: Matplotlib axis object.
    colors: the colors of nodes, blue or red
    average_right: the fraction of neighbors that share the same identity and the average thereof for right
    average_left: the fraction of neighbors that share the same identity and the average thereof for left
    iterations: the iteration in the network
    alterations: the amount of alterations in the network
    """
    ax.clear()

    frames_per_alteration = 15
    alpha = frame / frames_per_alteration  

    # clear existing edges and add new ones
    graph.clear_edges()
    for connection in network2:
        graph.add_edge(connection[0].ID, connection[1].ID)

    # update target positions based on new structure
    pos_target.update(nx.kamada_kawai_layout(graph, scale=0.8))

    # interpolate positions smoothly
    for node in pos:
        pos[node] = (1 - alpha) * np.array(pos[node]) + alpha * np.array(pos_target[node])

    # draw graph
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors, node_size=400)
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=graph.edges, edge_color="lightgray", width=0.5)

    # remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


    # add title
    ax.text(
    0.94, 1, 
    f"ITERATIONS   {iterations}", 
    fontsize=16, 
    color="gray", 
    transform=ax.transAxes, 
    ha="left", 
    va="center", 
    fontfamily="Arial"  
    )

    ax.text(
    0.94, 0.96, 
    f"ALTERATIONS  {alterations}", 
    fontsize=16, 
    color="gray", 
    transform=ax.transAxes, 
    ha="left", 
    va="center", 
    fontfamily="Arial" 
    )

    # red circle for right average
    ax.add_patch(Circle((0.96, 0.91), 0.02, color="#FF6666", transform=ax.transAxes))
    text_right = ax.text(
        0.99, 0.91, f"           {average_right:.2f}",
        fontsize=16, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial"
    )

    # blue circle for left average
    ax.add_patch(Circle((0.96, 0.85), 0.02, color="lightblue", transform=ax.transAxes))
    text_left = ax.text(
        0.99, 0.85, f"           {average_left:.2f}",
        fontsize=16, color="gray", transform=ax.transAxes, ha="left", va="center", fontfamily="Arial"
    )

def merging_gif(network1, network2, nodes_amount_left, nodes_amount_right, i, average_right, average_left, iterations, alterations):
    """
    To merge to cascade gifs we sometimes need a merging gif if the network has changed. This is what this is for.

    Args:
        network1: the network before an alteration
        network2: the network aftera an alteration
        nodes_amount_left
        nodes_amount_right
        average_left: the fraction of neighbors that share the same identity and the average thereof for left
        average_right: the fraction of neighbors that share the same identity and the average thereof for right
        iterations: the iteration in the network
        alterations: the amount of alterations in the network
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = ['lightblue'] * nodes_amount_left + ['#FF6666'] * nodes_amount_right
    graph = nx.Graph()
    graph.add_nodes_from(range(nodes_amount_left + nodes_amount_right))

    for connection in network1:
        graph.add_edge(connection[0].ID, connection[1].ID)
    
    pos = nx.kamada_kawai_layout(graph, scale=0.6)
    pos_target = pos.copy()  

    for spine in ax.spines.values(): 
        spine.set_visible(False)

    ani = FuncAnimation(
        fig, 
        update_for_gif, 
        frames=15, 
        interval=1000/15,
        fargs=(graph, network1, network2, pos, pos_target, ax, colors, average_right, average_left, iterations, alterations)
    )
    
    ani.save(f"animations/merging{i}.gif", fps=15, writer="ffmpeg")
    plt.show()

def merge_gifs(gif1_path, gif2_path, output_path):
    """
    merge two gifs

    Args: 
        gif1_path: path of first gif
        gif2_path: path of second gif
        output_path: output path of final gif

    """
    # open the first GIF
    gif1 = Image.open(gif1_path)
    frames1 = [frame.copy() for frame in ImageSequence.Iterator(gif1)]
    durations1 = [frame.info.get("duration", 100) for frame in ImageSequence.Iterator(gif1)]

    # open the second GIF
    gif2 = Image.open(gif2_path)
    frames2 = [frame.copy() for frame in ImageSequence.Iterator(gif2)]
    durations2 = [frame.info.get("duration", 100) for frame in ImageSequence.Iterator(gif2)]

    # combine frames and durations
    merged_frames = frames1 + frames2
    merged_durations = durations1 + durations2

    # save the merged GIF with correct durations
    merged_frames[0].save(
        output_path,
        save_all=True,
        append_images=merged_frames[1:],
        duration=merged_durations,  
        loop=0 
    )

def create_animation(network, cascade_amount):
    """
    Create an amazing animation of several cascades

    Args:
        network: the network object
        cascade_amount: the amount of cascades you want in the animation
    """
    for i in range(cascade_amount):
        cascade_gif_name = f"animations/cascade{i}.gif"
        merging_gif_name = f"animations/merging{i}.gif"
        output_name = f"animations/full_cascade{i}.gif"
        old_network_connections, connections, length_left, length_right, change, average_right, average_left, iterations, alterations = plot_cascade(network, i)
        if change:
            merging_gif(old_network_connections, connections, length_left, length_right, i, average_right, average_left, iterations, alterations)
            merge_gifs(cascade_gif_name, merging_gif_name, output_name)
        else:
            old_name = f"animations/cascade{i}.gif"
            new_name = f"animations/full_cascade{i}.gif"
            
            if os.path.exists(old_name): 
                os.rename(old_name, new_name)
            else:
                print(f"File {old_name} doesn't exist")

    for i in range(cascade_amount):
        if i != cascade_amount - 1:
            first = f"animations/full_cascade{i}.gif"
            second = f"animations/full_cascade{i+1}.gif" 
            merge_gifs(first, second, second)

def create_complete_animation(network, cascade_amount):
    """
    This creates an animation that merges the cascade animations and the plot network animation

    Args:
        network: the network object
        cascade_amount: this is the amount of cascades that are in the previous gif of all the cascades
    """
    input_file = f"animations/full_cascade{cascade_amount - 1}.gif"
    output_file = "animations/holy_grale.gif"
    merging_gif_name = f"animations/network_animation{cascade_amount}.gif"
    plot_network(network, cascade_amount)
    merge_gifs(input_file, merging_gif_name, output_file)

def plot_cascades_gamma(cas, num_runs, what_net):
    '''
    This plot calculates the average cascade size and polarization per correlation value for two networks.
    allowing for clear visual comparison

    args: 
    cas-cascade distributions for both networks (along with polarization)
    num_runs-number of experiments.
    what_net- "random" (comparing before and after for random), "scale_free" (comparing before after scale_free) 
    "both" (comparing after scale free vs after random)
    '''
    values_bef, values_af, variance_bef, variance_af = calculate_average_per_gamma(cas, num_runs)
    fig, ax = plt.subplots(figsize=(5,4))

    green_to_red = plt.cm.RdYlGn_r
    # Normalize polarization for coloring
    keys_bef = list(values_bef.keys())  # Gamma values (X-axis)
    sizes_bef = [v[0] for v in values_bef.values()]  # Sizes for Y-axis
    pol_bef = [v[1] for v in values_bef.values()]  # Polarization for colors

    keys_af = list(values_af.keys())  # Gamma values (X-axis)
    sizes_af = [v[0] for v in values_af.values()]  # Sizes for Y-axis
    pol_af = [v[1] for v in values_af.values()]  # Polarization for colors

    #Coloring scheme with polarization
    norm = plt.Normalize(vmin=0, vmax=1)
    colors_bef = [green_to_red(norm(p)) for p in pol_bef]
    colors_af = [green_to_red(norm(p)) for p in pol_af]
    
    var_sizes_bef = np.array([variance_bef[k][0] for k in keys_bef])  # Variance for sizes (Before)
    var_sizes_af = np.array([variance_af[k][0] for k in keys_af])  # Variance for sizes (After)

    #Computing confidence interval with calculated variance
    sem_sizes_bef = np.sqrt(var_sizes_bef) / np.sqrt(num_runs)
    sem_sizes_af = np.sqrt(var_sizes_af) / np.sqrt(num_runs)
    ci_sizes_bef = 1.96 * sem_sizes_bef
    ci_sizes_af = 1.96 * sem_sizes_af

    sm = plt.cm.ScalarMappable(cmap=green_to_red, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("0:non polarized-1:fully polarized")
    
    
    # make an errorbar to visualize CI
    ax.errorbar(keys_bef, sizes_bef, yerr=ci_sizes_bef, color="black", linewidth = 0.8, capsize=2, label="", linestyle="none", zorder=1)
    ax.errorbar(keys_af, sizes_af, yerr=ci_sizes_af, color="black", linewidth=0.8, capsize=2, label="", linestyle="none", zorder=1)

    # distinct between before after comparison and scale-free vs random comparison
    if what_net != "both":
        label1 = "Before"
        label2 = "After"
        marker1 = "o"
        marker2 = "s"
    else:
        label1 = "Random"
        label2 = "Scale-free"
        marker1 = "v"
        marker2 = "h"

    # plotting for both networks
    for x, y, color in zip(keys_bef, sizes_bef, colors_bef):
        ax.scatter(x, y, color=color, edgecolor="black", linewidths=0.5, marker=marker1, s=50, label="Before" if x == keys_bef[0] else "", zorder=2)

    for x, y, color in zip(keys_af, sizes_af, colors_af):
        ax.scatter(x, y, color=color, edgecolor="black", linewidths=0.5, marker=marker2, s=50, label="After" if x == keys_af[0] else "", zorder=2)
    
    
    legend_handles = [
    plt.Line2D([], [], marker=marker1, color="w", markerfacecolor="white", markersize=6, markeredgecolor="black", label=label1),
    plt.Line2D([], [], marker=marker2, color="w", markerfacecolor="white", markersize=6, markeredgecolor="black", label=label2),
    ]
    ax.set_xlabel("News Correlation")
    ax.set_ylabel("Average Cascade Size")
    ax.legend(handles=legend_handles)

    # saving plot in destined folder
    plt.savefig(f"plots/experiment_results/cascade_distribution/{what_net}/averaged_over_gammas.png", dpi=300, bbox_inches='tight')
    plt.show()

def test_significance(values_bef, values_af, variance_bef, variance_af, num_runs=30):
    """
    Perform t-tests on cascade sizes and polarization before vs. after for each gamma value.

    Args:
        values_bef : Dictionary mapping gamma values to (mean_size, mean_polarization) for before.
        values_af : Dictionary mapping gamma values to (mean_size, mean_polarization) for after.
        variance_bef : Dictionary mapping gamma values to (variance_size, variance_polarization) for before.
        variance_af : Dictionary mapping gamma values to (variance_size, variance_polarization) for after.
        num_runs : Number of runs per gamma value (default is 30).

    Returns:
        results : Dictionary mapping gamma values to t-test results for size and polarization.
    """
    results = {}

    for gamma in values_bef.keys():
        mean_size_bef, mean_pol_bef = values_bef[gamma]
        mean_size_af, mean_pol_af = values_af[gamma]
        
        var_size_bef, var_pol_bef = variance_bef[gamma]
        var_size_af, var_pol_af = variance_af[gamma]

        # Compute t-test for cascade sizes
        t_size, p_size = ttest_ind_from_stats(mean1=mean_size_bef, std1=np.sqrt(var_size_bef), nobs1=num_runs,
                                              mean2=mean_size_af, std2=np.sqrt(var_size_af), nobs2=num_runs, 
                                              equal_var=False) 

        # Compute t-test for polarization
        t_pol, p_pol = ttest_ind_from_stats(mean1=mean_pol_bef, std1=np.sqrt(var_pol_bef), nobs1=num_runs,
                                            mean2=mean_pol_af, std2=np.sqrt(var_pol_af), nobs2=num_runs, 
                                            equal_var=False)  

        results[gamma] = {
            "t_size": t_size, "p_size": p_size,
            "t_pol": t_pol, "p_pol": p_pol
        }

    return results


def statistics_cascades(cas_sf, cas_rand, num_runs):
    cascades_before_averaged_sf, cascades_after_averaged_sf = cas_sf
    cascades_before_averaged_rand, cascades_after_averaged_rand = cas_rand
    which_cas = [(cascades_before_averaged_sf, cascades_after_averaged_sf), (cascades_before_averaged_rand, cascades_after_averaged_rand), (cascades_after_averaged_rand, cascades_after_averaged_sf)]

    for i, what in enumerate(["scale_free", "random", "both"]):
        values_bef, values_af, variance_bef, variance_af = calculate_average_per_gamma(which_cas[i], num_runs)
        results = test_significance(values_bef, values_af, variance_bef, variance_af, num_runs)
        output_file = f"statistics/dummy/cascades/results_bef_af_{what}.txt"

        with open(output_file, "w") as f:
            f.write(f"Statistical significance for {what} network type (cascade experiments)\n")
            for gamma, res in results.items():
                f.write(f"Gamma = {gamma}:\n")
                f.write(f"  Size: t = {res['t_size']:.3f}, p = {res['p_size']:.3g}\n")
                f.write(f"  Polarization: t = {res['t_pol']:.3f}, p = {res['p_pol']:.3g}\n")
                f.write("--------------------------------------------------------\n")