from src.classes.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import os
import time

def execute_experiment(num_processes, parameters):
    """Initializes a pool of threads to run the simulation in parallel.

    Args:
        num_processes (int): number of threads, must be less or equal to the number of cores of your machine
        parameters (list): list of parameters for the network and experiment

    Returns:
        array: gathered results form all threads
    """    
    assert num_processes <= os.cpu_count(), "Lower the number of processes (PROCESSES)"
    with Pool(num_processes) as pool:
        print(f"Starting parallel execution for {calculate_assortativity} schedule")
        results = pool.starmap(calculate_assortativity, parameters)

    return results

def plot_results(x_data, y_data, std):
    """Plot the calculated data."""
    plt.figure(figsize=(7, 5))
    plt.scatter(x_data, y_data, color='blue')
    plt.fill_between(x_data, y_data - std, y_data + std, color='blue', alpha=0.5)
    plt.xlabel(r'Information Ecosystem $\gamma$', fontsize=14)
    plt.ylabel('Political Assortativity', fontsize=14)
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

def calculate_assortativity(network_type, steady_state_iter, num_nodes, correlations, update_fraction, starting_distribution, p):
    """Calculate assortativity coefficient based on different correlations and network parameters.

    Args:
        network_type (str): type of network (random or scale-free)
        steady_state_iter (int): number of update rounds to reach a steady state
        num_nodes (int): number of nodes in the network
        correlations (list): list of different news correlations
        update_fraction (float): fraction of nodes that directly sample the news
        starting_distribution (float): fraction of nodes L (or R)
        p (float, optional): probability to create an edge. Defaults to 0.1.

    Returns:
        arrays containing the mean assortativity coefficient and the corresponding standard deviation
    """    
    assort_coefs = np.zeros(len(correlations))

    for j, corr in enumerate(correlations):

        network = Network(network_type, num_nodes, mean=0, correlation=corr, starting_distribution=starting_distribution, 
                          update_fraction=update_fraction, p=p)

        for k in range(int(steady_state_iter)):
            network.update_round()

        nx_network = convert_to_nx(network)

        assort_coef = nx.attribute_assortativity_coefficient(nx_network, "ideology")
        assort_coefs[j] = assort_coef

    return assort_coefs

def run_experiment(num_threads, num_runs, network_type, steady_state_iter, num_nodes, correlations, update_fraction, starting_distribution, p):
    """Runs the experiment for the assortativity coefficient in parallel.

    Args:
        num_threads (int): number of threads used
        num_runs (int): number of runs 
        network_type (str): type of network
        steady_state_iter (int): number of update rounds to achieve a steady state
        num_nodes (int): number of nodes in the network
        correlations (list): list of tested correlation values
        update_fraction (float): fraction of nodes that directly sample the news
        starting_distribution (float): fraction of nodes of identity L (or R)
        p (float): probability to create an edge in a random network

    Returns:
        array: arrays containing the mean and standard deviation (p = 95%) of the assortativity coefficient  
    """    
    parameter_list = [
        (network_type, steady_state_iter, num_nodes, correlations, update_fraction, starting_distribution, p)
        for _ in range(num_runs)
    ]

    start = time.time()
    results = execute_experiment(num_threads, parameter_list)
    stop = time.time()
    print(f'Duration: {(stop-start)/60} min')

    data = np.vstack(results)
    mean = np.mean(data, axis=0)
    std = 1.96 * np.std(data, axis=0) / np.sqrt(num_runs) # confidence intervals at p = 95% confidence level
    plot_results(correlations, mean, std)

    return mean, std


# if __name__ == '__main__':
    # num_nodes = 100
    # correlations = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    # update_fraction = 0.1
    # starting_distribution = 0.5
    # p = 0.04
    # num_runs = 14
    # steady_state_iter = 1e4
    # network_type = 'random'

    # parameter_list = [
    #     (network_type, steady_state_iter, num_nodes, correlations, update_fraction, starting_distribution, p)
    #     for _ in range(num_runs)
    # ]

    # start = time.time()
    # results = execute_experiment(14, parameter_list)
    # stop = time.time()
    # print(f'Duration: {(stop-start)/60} min')

    # data = np.vstack(results)
    # mean = np.mean(data, axis=0)
    # std = 1.96 * np.std(data, axis=0) / np.sqrt(num_runs) # confidence intervals at p = 95% confidence level
    # plot_results(correlations, mean, std)
