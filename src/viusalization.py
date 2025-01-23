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

def self_sort(frame, network, graph, colors, pos, pos_target, ax, seedje):
    ax.clear()
    for _ in range(10000):
        network.update_round()
    graph.clear_edges()
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)
    
    if frame % 100 == 0:
        seedje+=frame
        pos_target.update(nx.spring_layout(graph, k=0.4, iterations=50, seed=seedje))  # Update target positions

    # Smoothly interpolate positions between frames
    alpha = (frame % 100) / 1000
    for node in pos:
        pos[node] = pos[node] * (1 - alpha) + pos_target[node] * alpha 
    
    # print(list(graph.nodes()))
    nx.draw(
        graph, 
        pos,
        ax=ax,
        with_labels=False, 
        node_color= colors, 
        node_size=200, 
        font_size=12, 
        # font_weight="bold", 
        edge_color="gray")
    ax.set_title(f"frame: {frame}", fontsize=14)

def plot_network(network):
    fig, ax = plt.subplots(figsize=(6, 6))
    graph, colors = initial_graph(network)
    seedje = 33
    pos = nx.spring_layout(graph, k=0.4, iterations=25, seed=seedje)
    pos_target = pos.copy()
    # Create the animation
    ani = FuncAnimation(fig, 
                        self_sort, 
                        frames=1000, 
                        interval=500,  
                        fargs=(network, graph, colors, pos, pos_target, ax, seedje))
    ani.save("animations/network_animation.gif", fps=10, writer="pillow")
    plt.show()