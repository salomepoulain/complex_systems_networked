from src.classes.network import Network
import numpy as np
import matplotlib.pyplot as plt

num_nodes = 100
correlation = -1
update_fraction = 0.1
starting_distribution = 0.5
p = 0.1
# k = 2
num_runs = 10
steady_state_iter = 1e4

def average_degree_dist(num_runs, steady_state_iter, num_nodes, correlation, update_fraction, starting_distribution, p=0.1, k=None):
    """Calculates averaged degree distribution for multiple networks."""
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

degrees, freqs = average_degree_dist(num_runs, steady_state_iter, num_nodes, correlation, update_fraction, starting_distribution, p)

plt.bar(degrees, freqs, color='skyblue', edgecolor='black')
plt.xlabel('Degree', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(degrees) 
plt.show()