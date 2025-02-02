import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

def plot_results(x_data, mean_same, std_same, mean_diff, std_diff):
    """Plot the change in social ties."""
    plt.figure(figsize=(6,5), dpi=500)
    plt.scatter(x_data, mean_same, color='navy', label='Same Ideology')
    plt.fill_between(x_data, mean_same - std_same, mean_same + std_same, color='navy', alpha=0.5)
    plt.scatter(x_data, mean_diff, color='royalblue', label='Different Ideology')
    plt.fill_between(x_data, mean_diff - std_diff, mean_diff + std_diff, color='royalblue', alpha=0.5)
    plt.hlines(0, xmin=-1.1, xmax=1.1, color='grey', ls='dotted')
    plt.xlabel('News Correlation', fontsize=14)
    plt.ylabel(r'Net $\Delta$ social ties', fontsize=14)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-2, 2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def gather_social_ties(network):
    """Gathers the number of L and R connections for all nodes.

    Args:
        network : generated network

    Returns:
        dict: contains node ID, political identity, and number of L and R connections.
    """    
    social_ties = {}

    for node in network.all_nodes:
        identity = 1 if node.identity == 'L' else 0

        connection_L = sum(1 for connection in node.node_connections if connection.identity == 'L')
        connection_R = sum(1 for connection in node.node_connections if connection.identity == 'R')
        
        social_ties[node.ID] = (identity, connection_L, connection_R)
    
    return social_ties

def calculate_net_gain_loss(initial, final):
    """Calculates the difference in L and R connections after the update rounds.

    Args:
        initial (dict): gathered social ties data from the initial network
        final (dict): gathered social ties data from the final

    Returns:
        tuple: average net social ties for different and opposing ideologies over all nodes
    """ 
    diff_ideologies = []
    same_ideologies = []
    for node_ID, (identity, init_L, init_R) in initial.items():
        final_L, final_R = final[node_ID][1], final[node_ID][2]
        if identity == 1:
            same_ideology = final_L - init_L
            diff_ideology = final_R - init_R
        else: 
            same_ideology = final_R - init_R
            diff_ideology = final_L - init_L
        
        diff_ideologies.append(diff_ideology)
        same_ideologies.append(same_ideology)

    return np.mean(same_ideologies), np.mean(diff_ideologies)

def run_social_ties_experiment(networks, network_type, num_runs, save_results=False):
    """Calculates and plots the net change in social ties between the initial and final network as a function of the news correlation.

    Args:
        networks (dict): all generated networks
        network_type (str): type of network
        num_runs (int): number of repetitions
        save_results (bool, optional): option to save data to csv file. Defaults to False.
    """    
    all_data_same = defaultdict(list)
    all_data_diff = defaultdict(list)
    for (corr, run), (init_nw, final_nw) in networks.items():

        init_social_ties = gather_social_ties(init_nw)
        final_social_ties = gather_social_ties(final_nw)
        
        net_same_ideology, net_diff_ideology = calculate_net_gain_loss(init_social_ties, final_social_ties)

        all_data_same[corr].append(net_same_ideology)
        all_data_diff[corr].append(net_diff_ideology)
    
    correlations = sorted(all_data_same.keys())
    mean_same = np.array([np.mean(all_data_same[corr]) for corr in correlations])
    conf_int_same = np.array([1.96*np.std(all_data_same[corr])/np.sqrt(num_runs) for corr in correlations])
    mean_diff = np.array([np.mean(all_data_diff[corr]) for corr in correlations])
    conf_int_diff = np.array([1.96*np.std(all_data_diff[corr])/np.sqrt(num_runs) for corr in correlations])

    if save_results:
        dict_results = {
            "Correlations": correlations,
            "mean_same": mean_same,
            "std_same": conf_int_same,
            "mean_diff": mean_diff,
            "std_diff": conf_int_diff,
        }
        df = pd.DataFrame(dict_results)
        df.to_csv(f"social_ties_{network_type}_results.csv", index=False)
        print(f'Results are saved in: social_ties_{network_type}_results.csv')

    plot_results(correlations, mean_same, conf_int_same, mean_diff, conf_int_diff)
