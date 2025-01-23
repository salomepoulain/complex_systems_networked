# Calculates the assortativity coefficient of a graph. 

from src.classes.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def convert_to_nx(network):
    """Convert network to nx Graph to use built in functions.

    Args:
        network (set): network to be converted.

    Returns:
        networkX graph
    """    
    graph = nx.Graph()

    for node in network.all_nodes:
        graph.add_node(node.ID, ideology=node.identity)

    for connection in network.connections:
        graph.add_edge(connection[0].ID, connection[1].ID)

    return graph

def calculate_assortativity(num_runs, steady_state_iter, num_nodes, correlations, update_fraction, starting_distribution, p=0.1, k=None):
    """Calculate assortativity coefficient based on different correlations and network parameters.

    Args:
        num_runs (int): number of repetitions
        steady_state_iter (int): number of update rounds to reach a steady state
        num_nodes (int): number of nodes in the network
        correlations (list): list of different news correlations
        update_fraction (float): fraction of nodes that directly sample the news
        starting_distribution (float): fraction of nodes L (or R)
        p (float, optional): probability to create an edge. Defaults to 0.1.
        k (int, optional): number of edges per node. Defaults to None.

    Returns:
        arrays containing the mean assortativity coefficient and the corresponding standard deviation
    """    
    assort_coefs = np.zeros((num_runs, len(correlations)))

    for i in range(num_runs):

        for j, corr in enumerate(correlations):

            network = Network(num_nodes, corr, update_fraction, starting_distribution, p)

            for k in range(int(steady_state_iter)):
                network.update_round()

            nx_network = convert_to_nx(network)

            assort_coef = nx.attribute_assortativity_coefficient(nx_network, "ideology")

            assort_coefs[i, j] = assort_coef

    mean_assort_coef = np.mean(assort_coefs, axis=0)
    std_assort_coef = 1.96 * np.std(assort_coefs, axis=0) / np.sqrt(num_runs) # p=95% confidence interval

    return mean_assort_coef, std_assort_coef

num_nodes = 100
correlations = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
update_fraction = 0.1
starting_distribution = 0.5
p = 0.1
# k = 2
num_runs = 10
steady_state_iter = 1e4

mean_coef, std_coef = calculate_assortativity(num_runs, steady_state_iter, num_nodes, correlations, update_fraction, starting_distribution, p)

plt.figure(figsize=(7, 5))
plt.scatter(correlations, mean_coef, color='blue')
plt.fill_between(correlations, mean_coef - std_coef, mean_coef + std_coef, color='blue', alpha=0.5)
plt.xlabel(r'Information Ecosystem $\gamma$', fontsize=14)
plt.ylabel('Political Assortativity', fontsize=14)
plt.show()