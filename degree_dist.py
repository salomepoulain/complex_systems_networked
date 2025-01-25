from src.classes.network import Network
import numpy as np
import matplotlib.pyplot as plt

# Example parameters
num_nodes = 100
correlation = -1
update_fraction = 0.1
starting_distribution = 0.5
p = 0.1
num_runs = 10
steady_state_iter = 1e4

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


def degree_and_thrshld_correlation(steady_state_iter, num_nodes, correlation, update_fraction, starting_distribution, p=0.1, k=None):
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
    network = Network(num_nodes, 0, correlation, starting_distribution, update_fraction, p)

    for i in range(int(steady_state_iter)):
        network.update_round()

    degree_thrsh_values = np.zeros((2, len(network.all_nodes)))
    for j, node in enumerate(network.all_nodes):
        degree_thrsh_values[0, j] = len(node.node_connections)
        degree_thrsh_values[1, j] = node.response_threshold
    
    corr_coef = np.corrcoef(degree_thrsh_values[0], degree_thrsh_values[1])[0, 1]
    
    return degree_thrsh_values, corr_coef

def correlation_vs_updateround(num_runs, num_plot_points, max_rounds, num_nodes, correlation, update_fraction, starting_distribution, p):
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

        corr_per_round_number = np.zeros(num_runs)
        for j in range(num_runs):
            _, corr_coef = degree_and_thrshld_correlation(int(num_update), num_nodes, correlation, update_fraction, starting_distribution, p)
            corr_per_round_number[j] = corr_coef

        mean_corr = np.mean(corr_per_round_number)
        std_corr = 1.96 * np.std(corr_per_round_number) / np.sqrt(num_runs)

        all_mean_corr[i] = mean_corr
        all_std_corr[i] = std_corr

    return all_update_rounds, all_mean_corr, all_std_corr

update_rounds, mean_corr1, std_corr1 = correlation_vs_updateround(num_runs=20, num_plot_points=20, max_rounds=10000, num_nodes=100, 
                                                                correlation=-1, update_fraction=0.1, starting_distribution=0.5, p=0.1)
update_rounds, mean_corr2, std_corr2 = correlation_vs_updateround(num_runs=20, num_plot_points=20, max_rounds=10000, num_nodes=100, 
                                                                correlation=0, update_fraction=0.1, starting_distribution=0.5, p=0.1)
update_rounds, mean_corr3, std_corr3 = correlation_vs_updateround(num_runs=20, num_plot_points=20, max_rounds=10000, num_nodes=100, 
                                                                correlation=1, update_fraction=0.1, starting_distribution=0.5, p=0.1)
plt.figure(figsize=(10,7))
plt.scatter(update_rounds, mean_corr1, color='red', label=r'$\gamma = -1$')
plt.fill_between(update_rounds, mean_corr1 - std_corr1, mean_corr1 + std_corr1, color='red', alpha=0.5)
plt.scatter(update_rounds, mean_corr2, color='blue', label=r'$\gamma = 0$')
plt.fill_between(update_rounds, mean_corr2 - std_corr2, mean_corr2 + std_corr2, color='blue', alpha=0.5)
plt.scatter(update_rounds, mean_corr3, color='green', label=r'$\gamma = 1$')
plt.fill_between(update_rounds, mean_corr3 - std_corr3, mean_corr3 + std_corr3, color='green', alpha=0.5)
plt.xlabel('Number of update rounds', fontsize=14)
plt.ylabel('Correlation', fontsize=14)
plt.legend()
plt.show()



# degree_thrsh_values, corr_coef = degree_and_thrshld_correlation(steady_state_iter, num_nodes, correlation, update_fraction, starting_distribution, p)
# plt.figure(figsize=(7,5))
# plt.title(f'Correlation between degree and threshold: {round(corr_coef, 3)}')
# plt.scatter(degree_thrsh_values[0], degree_thrsh_values[1], color='blue')
# plt.xlabel(r'Node Degree $k$', fontsize=14)
# plt.ylabel(r'Node Threshold $\theta$', fontsize=14)
# plt.show()

# degrees, freqs = average_degree_dist(num_runs, steady_state_iter, num_nodes, correlation, update_fraction, starting_distribution, p)
# plt.bar(degrees, freqs, color='skyblue', edgecolor='black')
# plt.xlabel('Degree', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
# plt.xticks(degrees) 
# plt.show()