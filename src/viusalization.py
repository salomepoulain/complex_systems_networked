from src.classes.network import Network
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np

def initial_graph(network):
    colors = ['skyblue'] * len(network.nodesL) + ['red'] * len(network.nodesR)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(network.all_nodes)))
    return graph, colors

def self_sort(frame, network, graph, colors, pos, ax):
    ax.clear()
    network.update_round()
    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    
    # print(list(graph.nodes()))
    nx.draw(
        graph, 
        pos,
        ax=ax,
        with_labels=True, 
        node_color= colors, 
        node_size=1000, 
        font_size=12, 
        font_weight="bold", 
        edge_color="gray")
    ax.set_title(f"frame: {frame}", fontsize=14)

def plot_network(network):
    fig, ax = plt.subplots(figsize=(6, 6))
    graph, colors = initial_graph(network)

    pos = nx.spring_layout(graph, k=0.5, iterations=5, seed=21)
    # Create the animation
    ani = FuncAnimation(fig, 
                        self_sort, 
                        frames=500, 
                        interval=500,  
                        fargs=(network, graph, colors, pos, ax))
    ani.save("network_animation.gif", fps=10, writer="pillow")
    plt.show()