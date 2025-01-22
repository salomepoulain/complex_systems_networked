from src.classes.network import Network
import matplotlib.pyplot as plt
import networkx as nx

def plot_network(network):
    graph = nx.Graph()
    plt.figure(figsize=(6,6))
    for node in network.all_nodes:
        graph.add_node(node.ID)
    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)

    pos = nx.spring_layout(graph)  # Layout for positioning the nodes
    nx.draw(graph, pos, with_labels=True, node_color="skyblue", node_size=1000, font_size=12, font_weight="bold", edge_color="gray")
    plt.title("Custom Network Visualization", fontsize=14)
    plt.show()