from src.classes.network import Network
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import numpy as np

def get_network_properties(network, seed):
    # Replace with actual calculations for your network
    corr = network.correlation
    node_info = []
    connection_IDs = []
    for node in network.all_nodes:
        node_info.append((node.ID, node.identity, node.response_threshold))
    for conn in network.connections:
        connection_IDs.append((conn[0].ID, conn[1].ID))
    properties = {
        "Number of Nodes": len(network.all_nodes),
        "Number of Edges": len(network.connections),
        "Correlation": corr,
        "P value": network.p,
        "Seed": seed,
        "Update fraction": network.update_fraction,
        "Connections": connection_IDs,
        "Nodes": node_info
    }
    return properties


def parallel_network_generation(whichrun, num_nodes, seed, corr, iterations, update_fraction, starting_distribution, p, m=0, network_type="random"):
    seed+=whichrun
        # average degree of 8
    if network_type == "random":
        network = Network(network_type, num_nodes, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seed, p=p)

    output_folder = f"networks/{network_type}/{corr}" 
    output_filename = f"network_{whichrun}.txt"  
    output_path = os.path.join(output_folder, output_filename)
    os.makedirs(output_folder, exist_ok=True)
    number_of_alterations = 0

    for _ in range(iterations):
        network.update_round()
        number_of_alterations += network.alterations
        network.clean_network()
    print(number_of_alterations)
    
    # Get network properties
    network_properties = get_network_properties(network, seed)

    # Write the properties to the file
    with open(output_path, "w") as file:
        file.write("Network Properties\n")
        file.write("==================\n")
        for key, value in network_properties.items():
            file.write(f"{key}: {value}\n")


def generate_networks(correlations, initial_seeds, num_nodes, iterations, how_many, update_fraction, starting_distribution, p):
    runs = np.arange(how_many)  # Create a range for the runs
    num_threads = min(how_many, 10)
    for j,corr in enumerate(correlations): 
        seed = int(initial_seeds[j])
        num_threads = 10
        
        worker_function = partial(parallel_network_generation, num_nodes=num_nodes, seed=seed, corr=corr, iterations=iterations, 
                                  update_fraction=update_fraction, starting_distribution=starting_distribution, p=p, m=0, network_type="random")
        with ProcessPoolExecutor(max_workers=num_threads) as executer:
            list(executer.map(worker_function, runs))
