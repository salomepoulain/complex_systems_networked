from src.classes.network import Network
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
import os

def average_degree_dist(num_runs, steady_state_iter, num_nodes, correlation, update_fraction, starting_distribution, p=0.1, k=None):
    """Calculates averaged degree distribution for multiple networks.

    Args:
        num_runs (int): number of replications, e.g., number of networks
        steady_state_iter (int): number of simulation rounds before a steady state is reached
        num_nodes (int): total number of nodes in the network
        correlation (float): correlation between the news sources
        update_fraction (float): fraction of nodes that sample directly from the news
        starting_distribution (float): fraction of nodes with identity L (or R)
        p (float, optional): probability to create edge. Defaults to 0.1.
        k (int, optional): number of connections for each node. Defaults to None.

    Returns:
        Arrays containing the unique degrees and the average frequency.
    """    
    degree_all = set()
    total_degrees = None

    for j in range(num_runs):

        # Create network and let run for some amount of runs
        network = Network(num_nodes, 0, correlation, starting_distribution, update_fraction, p)
        for i in range(int(steady_state_iter)):
            network.update_round()

        # Calculate the degree distribution for this network
        degree_list = [len(node.connections) for node in network.all_nodes]
        degrees, freqs = np.unique(degree_list, return_counts=True)
        degree_freq_dict = dict(zip(degrees, freqs))
        degree_all.update(degree_freq_dict.keys())

        # Add the network to the other networks
        if total_degrees is None:
            total_degrees = degree_freq_dict
        else:
            for degree in degree_freq_dict:
                total_degrees[degree] = total_degrees.get(degree, 0) + degree_freq_dict[degree]

    # Calculate the average over all degree distributions
    avg_degrees = {}
    for degree in degree_all:
        avg_degrees[degree] = total_degrees.get(degree, 0) / num_runs

    # Convert back to array for usability
    degrees = np.array(sorted(avg_degrees.keys()))
    avg_freqs = np.array([avg_degrees[d] for d in degrees])

    return degrees, avg_freqs

def plot_data(data, correlations):
    """Plot the simulated data."""
    colors = ['red', 'blue', 'green']
    plt.figure(figsize=(10,7))
    for i in range(len(data)):
        plt.scatter(data[i, 0], data[i, 1], color=colors[i], label=fr'$\gamma = {{{correlations[i]}}}$')
        plt.fill_between(data[i, 0], data[i, 1] - data[i, 2], data[i, 1] + data[i, 2], color=colors[i], alpha=0.5)

    plt.xlabel('Number of update rounds', fontsize=14)
    plt.ylabel('Correlation', fontsize=14)
    plt.legend()
    plt.show()

def degree_and_thrshld_correlation(network_type, steady_state_iter, num_nodes, correlation, update_fraction, starting_distribution, p=0.1, k=None):
    """Calculates the correlation coefficient between node degree and node threshold.

    Args:
        steady_state_iter (int): number of simulation rounds before a steady state is reached
        num_nodes (int): total number of nodes in the network
        correlation (float): correlation between the news sources
        update_fraction (float): fraction of nodes that sample directly from the news
        starting_distribution (float): fraction of nodes with identity L (or R)
        p (float, optional): probability to create edge. Defaults to 0.1.
        k (int, optional): number of connections for each node. Defaults to None.

    Returns:
        Tuple of an array containing the degree and threshold, and the correlation coefficient.
    """    
    network = Network(network_type, num_nodes, mean=0, correlation=correlation, starting_distribution=starting_distribution, 
                      update_fraction=update_fraction, p=p)

    for i in range(int(steady_state_iter)):
        network.update_round()

    degree_thrsh_values = np.zeros((2, len(network.all_nodes)))
    for j, node in enumerate(network.all_nodes):
        degree_thrsh_values[0, j] = len(node.node_connections)
        degree_thrsh_values[1, j] = node.response_threshold
    
    corr_coef = np.corrcoef(degree_thrsh_values[0], degree_thrsh_values[1])[0, 1]
    
    return degree_thrsh_values, corr_coef

def correlation_vs_updateround(num_threads, num_runs, num_plot_points, max_rounds, network_type, num_nodes, correlation, update_fraction, starting_distribution, p):
    """Calculates the correlation between degree and threshold for different amounts of update rounds.

    Args:
        num_runs (int): number of repetitions for each number of rounds
        num_plot_points (int): number of points to calculate and plot
        max_rounds (int): maximum number of update rounds
        num_nodes (int): number of nodes in the network
        correlation (float): correlation between the news sources
        update_fraction (float): fraction of nodes that sample directly from the news
        starting_distribution (float): fraction of nodes with identity L (or R)
        p (float, optional): probability to create edge. Defaults to 0.1.

    Returns:
        arrays for mean correlations, standard deviations, and number of update rounds 
    """    
    all_update_rounds = np.linspace(1, max_rounds, num_plot_points)
    all_mean_corr = np.zeros(num_plot_points)
    all_std_corr = np.zeros(num_plot_points)
    for i, num_update in enumerate(all_update_rounds):

        worker_function = partial(degree_and_thrshld_correlation, network_type, int(num_update), num_nodes, correlation, 
                                  update_fraction, starting_distribution, p)

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(worker_function, range(num_runs)))
        
        corr_per_round_number = np.array([result[1] for result in results])

        mean_corr = np.mean(corr_per_round_number)
        std_corr = 1.96 * np.std(corr_per_round_number) / np.sqrt(num_runs)

        all_mean_corr[i] = mean_corr
        all_std_corr[i] = std_corr

    return all_update_rounds, all_mean_corr, all_std_corr

def run_degree_corr_experiment(num_threads, num_runs, num_plot_points, max_rounds, network_type, num_nodes, correlations, update_fraction, starting_distribution, p):
    """Run an experiment for the correlation between node degree and node threshold as a function of the number of update rounds. Multiple values of gamma can be used.

    Args:
        num_threads (int): number of threads used for parallel execution
        num_runs (int): number of repetitions
        num_plot_points (int): number of points to plot
        max_rounds (int): maximum number of update rounds
        network_type (str): type of network
        num_nodes (int): number of nodes in the network
        correlations (list): list of gamma values (correlation between news sources)
        update_fraction (float): fraction of nodes that directly sample from the news
        starting_distribution (float): fraction of nodes with identity L (or R)
        p (float): probability to create an edge in a random network
    """    
    assert num_threads <= os.cpu_count(), 'Num threads must be less or equal than your CPU count.'
    start = time.time()

    data_matrix = np.zeros((len(correlations), 3, num_plot_points))
    for i, corr in enumerate(correlations):
        update_rounds, mean_corr, std_corr = correlation_vs_updateround(num_threads, num_runs, num_plot_points, max_rounds, network_type, num_nodes, 
                                                                        corr, update_fraction, starting_distribution, p)
        data_matrix[i, 0, :] = update_rounds
        data_matrix[i, 1, :] = mean_corr
        data_matrix[i, 2, :] = std_corr

    stop = time.time()
    print(f"Duration: {(stop-start)/60} min")

    plot_data(data_matrix, correlations)


if __name__ == '__main__':
    run_degree_corr_experiment(num_threads=14, num_runs=56, num_plot_points=20, max_rounds=10000, network_type='random', num_nodes=100, 
                                                                    correlations=[-1, 0, 1], update_fraction=0.1, starting_distribution=0.5, p=0.05)
