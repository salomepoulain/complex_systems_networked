import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def plot_results(correlations, means, std):
    """Plot the calculated data."""
    plt.figure(figsize=(7, 5), dpi=300)
    plt.scatter(correlations, means, color='blue')
    plt.fill_between(correlations, means - std, means + std, color='blue', alpha=0.5)
    plt.xlabel(r'Information Ecosystem $\gamma$', fontsize=14)
    plt.ylabel('Political Assortativity', fontsize=14)
    plt.tight_layout()
    plt.show()


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


def run_assortativity_experiment(networks, network_type, num_runs, save_results=False):
    """Calculates and plots the assortativity as a function of news correlation.

    Args:
        networks (dict): all generated networks
        network_type (str): network type (random or scale_free)
        num_runs (int): number of repetitions
        save_results (bool, optional): save the calculated results. Defaults to False.

    """    
    assort_coefs = defaultdict(list)

    for (corr, run), (init_nw, final_nw) in networks.items():

        print(f'corr: {corr}, run: {run}')
        nx_network = convert_to_nx(final_nw)

        assort_coef = nx.attribute_assortativity_coefficient(nx_network, "ideology")
        assort_coefs[corr].append(assort_coef)
    
    correlations = sorted(assort_coefs.keys())
    means = np.array([np.mean(assort_coefs[corr]) for corr in correlations])
    conf_int = np.array([1.96 * np.std(assort_coefs[corr]) / np.sqrt(num_runs) for corr in correlations])

    if save_results:
        dict_results = {
            "correlations": correlations,
            "mean assort": means,
            "conf_int": conf_int,
        }
        df = pd.DataFrame(dict_results)
        df.to_csv(f"assortativity_{network_type}_results.csv", index=False)
        print(f'Results are saved in: assortativity_{network_type}_results.csv')

    plot_results(correlations, means, conf_int)


# Dummy data
correlations = np.linspace(-1, 1, 11)
correlations = np.round(correlations, 1)
initial_seeds = np.linspace(13, 1600, 11)
num_runs = 10
num_nodes = 200
update_fraction = 0.1
average_degree = 8
starting_distribution = 0.5
p = average_degree/(num_nodes-1) 
updates = 10000

# all_networks = read_and_load_networks(num_runs, num_nodes, update_fraction, average_degree, starting_distribution, correlations)
# run_assortativity_experiment(all_networks, 'random', num_runs, save_results=False)
