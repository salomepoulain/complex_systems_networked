from src.classes.network import Network
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_network(network):
    graph = nx.Graph()
    colors = ['skyblue'] * len(network.nodesL) + ['red'] * len(network.nodesR)
    plt.figure(figsize=(4,4))
    # print(network.all_nodes.ID)
    for node in network.all_nodes:
        graph.add_nodes_from(range(len(network.all_nodes)))
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)

    print(list(graph.nodes()))
    pos = nx.spring_layout(graph, k=0.5, iterations=5) 
    nx.draw(
        graph, 
        pos, 
        with_labels=True, 
        node_color= colors, 
        node_size=1000, 
        font_size=12, 
        font_weight="bold", 
        edge_color="gray")
    plt.title("Custom Network Visualization", fontsize=14)
    plt.show()
