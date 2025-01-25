from src.classes.network import Network
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
import os

def plot_results(x_data, mean_same, std_same, mean_diff, std_diff):
    """Plot the change in social ties."""
    plt.figure(figsize=(7,5))
    plt.scatter(x_data, mean_same, color='blue', label='Same Ideology')
    plt.fill_between(x_data, mean_same - std_same, mean_same + std_same, color='blue', alpha=0.5)
    plt.scatter(x_data, mean_diff, color='red', label='Different Ideology')
    plt.fill_between(x_data, mean_diff - std_diff, mean_diff + std_diff, color='red', alpha=0.5)
    plt.hlines(0, xmin=-1.1, xmax=1.1, color='grey', ls='dotted')
    plt.xlabel(r'Information Ecosystem $\gamma$', fontsize=14)
    plt.ylabel(r'Net $\Delta$ social ties', fontsize=14)
    plt.xlim(-1.1, 1.1)
    plt.legend()
    plt.show()

def gather_social_ties(network):
    """Gathers the number of L and R connections for all nodes.

    Args:
        network : generated network

    Returns:
        array: contains node ID, political identity, and number of L and R connections.
    """    
    social_ties = np.zeros((len(network.all_nodes), 4))

    for i, node in enumerate(network.all_nodes):
        identity = 1 if node.identity == 'L' else 0

        connection_L = sum(1 for connection in node.node_connections if connection.identity == 'L')
        connection_R = sum(1 for connection in node.node_connections if connection.identity == 'R')
        
        social_ties[i, :] = node.ID, identity, connection_L, connection_R
    
    return social_ties

def calculate_net_gain_loss(initial, final):
    """Calculates the difference in L and R connections after the update rounds.

    Args:
        initial (array): gathered social ties data from the initial network
        final (array): gathered social ties data from the final

    Returns:
        tuple: average net social ties for different and opposing ideologies over all nodes
    """    
    diff_ideologies = []
    same_ideologies = []
    for i in range(len(initial)):
        if final[i, 1] == 1:
            same_ideology = final[i, 2] - initial[i, 2]
            diff_ideology = final[i, 3] - initial[i, 3]
        else: 
            same_ideology = final[i, 3] - initial[i, 3]
            diff_ideology = final[i, 2] - initial[i, 2]
        
        diff_ideologies.append(diff_ideology)
        same_ideologies.append(same_ideology)

    return np.mean(same_ideologies), np.mean(diff_ideologies)

def calculate_net_social_ties(network_type, steady_state_iter, num_nodes, correlation, starting_distribution, update_fraction, p, dummy=None):
    """Creates the network and calculates the net social ties.

    Args:
        network_type (str): type of the network
        steady_state_iter (int): number of update rounds to reach a steady state
        num_nodes (int): number of nodes in the network
        correlation (float): correlation between the news sources
        starting_distribution (float): fraction of nodes with identity L or R
        update_fraction (float): fraction of nodes that directly sample the news
        p (float): probability of creating an edge
        dummy : dummy variable, ignore.

    Returns:
        tuple: average net social ties for different and opposing ideologies over all nodes
    """    
    network = Network(network=network_type, num_nodes=num_nodes, mean=0, correlation=correlation, 
                        starting_distribution=starting_distribution, update_fraction=update_fraction, p=p)
        
    initial_social_ties = gather_social_ties(network)

    for _ in range(steady_state_iter):
        network.update_round()

    final_social_ties = gather_social_ties(network)

    mean_same, mean_diff = calculate_net_gain_loss(initial_social_ties, final_social_ties)

    return mean_same, mean_diff

def social_ties_vs_ecosystem_parallel(num_threads, num_runs, network_type, steady_state_iter, num_nodes, correlations, starting_distribution, update_fraction, p):
    """Run the social ties experiment in parallel and for multiple values of the news correlation

    Args:
        num_threads (int): number of threads used
        num_runs (int): number of replications
        network_type (str): type of the network
        steady_state_iter (int): number of update rounds to reach a steady state
        num_nodes (int): number of nodes in the network
        correlation (float): correlation between the news sources
        starting_distribution (float): fraction of nodes with identity L or R
        update_fraction (float): fraction of nodes that directly sample the news
        p (float): probability of creating an edge

    Returns:
        array: contains all the data from the experiments (mean and standard deviation)
    """    
    all_data_same = np.zeros((len(correlations), 2))
    all_data_diff = np.zeros((len(correlations), 2))
    for i, corr in enumerate(correlations):

        worker_function = partial(calculate_net_social_ties, network_type, steady_state_iter, num_nodes, corr, starting_distribution, update_fraction, p)

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(worker_function, range(num_runs)))
        
        means_same = np.array([result[0] for result in results])
        means_diff = np.array([result[1] for result in results])

        all_data_same[i, :] = np.mean(means_same), 1.96*np.std(means_same)/np.sqrt(num_runs)
        all_data_diff[i, :] = np.mean(means_diff), 1.96*np.std(means_diff)/np.sqrt(num_runs)

    return all_data_same, all_data_diff

def run_social_ties_experiment(num_threads, num_runs, network_type, steady_state_iter, num_nodes, correlations, starting_distribution, update_fraction, p):
    """Runs the social ties experiment in parallel and plots the results.

    Args:
        num_threads (int): number of threads used
        num_runs (int): number of replications
        network_type (str): type of the network
        steady_state_iter (int): number of update rounds to reach a steady state
        num_nodes (int): number of nodes in the network
        correlation (float): correlation between the news sources
        starting_distribution (float): fraction of nodes with identity L or R
        update_fraction (float): fraction of nodes that directly sample the news
        p (float): probability of creating an edge
    """    
    assert num_threads <= os.cpu_count(), 'Num threads must be less or equal than your CPU count.'

    start = time.time()
    data_same, data_diff = social_ties_vs_ecosystem_parallel(num_threads, num_runs, network_type, steady_state_iter, num_nodes, correlations, starting_distribution, update_fraction, p)
    stop = time.time()
    print(f'Duration: {(stop-start)/60} min')

    plot_results(correlations, data_same[:, 0], data_same[:, 1], data_diff[:, 0], data_diff[:, 1])

if __name__ == '__main__':
    run_social_ties_experiment(num_threads=14, num_runs=56, network_type='random', steady_state_iter=10000, num_nodes=100, 
                               correlations=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
                                starting_distribution=0.5, update_fraction=0.1, p=0.05)
