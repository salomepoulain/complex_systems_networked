import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import os


def plot_results(correlations, means, std):
    """
    Plot the calculated data.
    
    Args:
        correlations (list): list of news correlations
        means (list): list of means
        std (list): list of standard deviations
    """
    plt.figure(figsize=(6, 5), dpi=500)
    plt.scatter(correlations, means, color='navy')
    plt.fill_between(correlations, means - std, means + std, color='navy', alpha=0.5)
    plt.xlabel('News Correlation', fontsize=14)
    plt.ylabel('Political Assortativity', fontsize=14)
    plt.ylim(-0.04, 0.25)
    plt.tight_layout()
    plt.show()


def assortativity_significance(save_results=False):
    """
    Determine if the difference between a random and scale-free network is statistically significant using the Welch T-test. 
    An FDR correction is used.
    
    Args:
        save_results (bool, optional): Save the results to a file. Defaults to False.

    """    
    data_random = pd.read_csv('statistics/assortativity/random_results.csv')
    data_scale = pd.read_csv('statistics/assortativity/scale_free_results.csv')

    random_means = data_random["mean assort"]
    random_std = data_random["std"]
    scale_means = data_scale["mean assort"]
    scale_std = data_scale["std"]
    n = 30

    # Compute Welch's t-test
    t_values = (random_means - scale_means) / np.sqrt((random_std**2 / n) + (scale_std**2 / n))
    p_values = 2 * (1 - norm.cdf(np.abs(t_values)))

    # Apply FDR correction
    p_adjusted = multipletests(p_values, method='fdr_bh')[1]

    results = []
    for i, corr in enumerate(data_random["correlations"]):
        result = f"Correlation {corr:.1f}: t = {t_values[i]:.3f}, p = {p_values[i]:.5f}, adjusted p = {p_adjusted[i]:.5f}"
        print(result)
        results.append(result)

    alpha = 0.05
    significant = data_random["correlations"][p_adjusted < alpha].values
    result_significant = f"\nSignificant differences (p < 0.05) at these correlations: \n{significant}"
    print(result_significant)
    results.append(result_significant)

    if save_results:
        os.makedirs(os.path.dirname("statistics/assortativity/statis_results.txt"), exist_ok=True)
        with open("statistics/assortativity/statis_results.txt", "w") as f:
            f.write("\n".join(results))
        print(f"Results saved to: statistics/assortativity/statis_results")


def convert_to_nx(network):
    """
    Convert network to nx Graph to use built in functions.

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


def run_assortativity_experiment(networks, network_type, num_runs, save_results=False, plot=True):
    """
    Calculates and plots the assortativity as a function of news correlation.

    Args:
        networks (dict): all generated networks
        network_type (str): network type (random or scale_free)
        num_runs (int): number of repetitions
        save_results (bool, optional): save the calculated results. Defaults to False.

    """    
    assort_coefs = defaultdict(list)

    for (corr, run), (init_nw, final_nw) in networks.items():
        nx_network = convert_to_nx(final_nw)

        assort_coef = nx.attribute_assortativity_coefficient(nx_network, "ideology")
        assort_coefs[corr].append(assort_coef)
    
    correlations = sorted(assort_coefs.keys())
    means = np.array([np.mean(assort_coefs[corr]) for corr in correlations])
    conf_int = np.array([1.96 * np.std(assort_coefs[corr]) / np.sqrt(num_runs) for corr in correlations])
    std = np.array([np.std(assort_coefs[corr]) for corr in correlations])

    if save_results:
        dict_results = {
            "correlations": correlations,
            "mean assort": means,
            "conf_int": conf_int,
            "std": std
        }
        df = pd.DataFrame(dict_results)
        df.to_csv(f"statistics/assortativity/{network_type}_results.csv", index=False)
        print(f'Results are saved in: assortativity_{network_type}_results.csv')

    if plot:
        plot_results(correlations, means, conf_int)