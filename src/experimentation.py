from src.classes.network import Network
from src.classes.node import Node
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import numpy as np
from collections import defaultdict

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

def generate_networks(correlations, initial_seeds, num_nodes, iterations, how_many, update_fraction, starting_distribution, p, network_sort="random"):
    print(f"starting parallel generation of {network_sort} networks ({num_nodes} nodes)")
    print("-----------------------------------------")
    runs = np.arange(how_many)  # Create a range for the runs
    num_threads = min(how_many, 10)
    for j,corr in enumerate(correlations): 
        print(f"starting correlation {corr}")
        seed = int(initial_seeds[j])
        num_threads = 10
        
        worker_function = partial(parallel_network_generation, num_nodes=num_nodes, seed=seed, corr=corr, iterations=iterations, 
                                  update_fraction=update_fraction, starting_distribution=starting_distribution, p=p, m=0, network_type=network_sort)
        with ProcessPoolExecutor(max_workers=num_threads) as executer:
            list(executer.map(worker_function, runs))


def read_network_properties(file_path):
    """
    Reads network properties from a .txt file and converts them back
    into a dictionary with appropriate datatypes.

    Args:
        file_path (str): Path to the .txt file containing network properties.

    Returns:
        dict: Network properties with restored data types.
    """
    properties = {}

    with open(file_path, "r") as file:
        lines = file.readlines()
    
    for line in lines[2:]:  # Skip the header lines
        key, value = line.strip().split(": ", 1)
        if key == "Number of Nodes" or key == "Number of Edges":
            properties[key] = int(value)
        elif key == "Correlation" or key == "P value" or key == "Update fraction":
            properties[key] = float(value)
        elif key == "Seed":
            properties[key] = int(value)
        elif key == "Connections":
            # Parse connections as a list of tuples
            connections = eval(value)  # Use eval to safely parse the list of tuples
            properties[key] = [(int(a), int(b)) for a, b in connections]
        elif key == "Nodes":
            # Parse nodes as a list of tuples
            nodes = eval(value)  # Use eval to safely parse the list of tuples
            properties[key] = [(int(node_id), identity, float(threshold)) for node_id, identity, threshold in nodes]
        else:
            properties[key] = value
    return properties

def read_and_load_networks(num_runs, num_nodes, update_fraction, average_degree, starting_distribution, correlations):
    p = average_degree/(num_nodes-1) 
    networks = defaultdict(tuple)
    for corr in correlations:
        for i in range(num_runs):
            network_properties = read_network_properties(f"networks/random/{corr}/network_{i}.txt")
            seedje = network_properties["Seed"]
            search_nodes = defaultdict(Node)
            before_network = Network("random", num_nodes, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seedje, p=p)
            after_network = Network("random", num_nodes, mean=0, correlation=corr, update_fraction=update_fraction, starting_distribution=starting_distribution, seed=seedje, p=p)
            after_network.connections = set()

            for nodeje in after_network.all_nodes:
                nodeje.node_connections = set()
                search_nodes[nodeje.ID] = nodeje
                
            for (node1, node2) in network_properties["Connections"]:
                search_nodes[node1].node_connections.add(search_nodes[node2])
                after_network.connections.add((search_nodes[node1], search_nodes[node2]))
            
            networks[(corr, i)] = (before_network, after_network)

    return networks


def create_data(iters, network):

    all_cascade_sizes = []
    all_polarizations = []
    average_cascade_per_round = []
    average_polarization_per_round = []
    number_of_samplers = 20

    for _ in range(iters): 
        cascades, cascade_dist, cascade_polarization = network.analyze_network()
        average_cascade_per_round.append(sum(cascade_dist)/number_of_samplers)
        average_polarization_per_round.append(sum(cascade_polarization))
        all_cascade_sizes += cascade_dist
        all_polarizations += cascade_polarization

        # plot_network(network, cascades)

    data = defaultdict(list)
    for i, (size, polarization) in enumerate(zip(all_cascade_sizes, all_polarizations), 1):
        data[size].append(polarization)
    for size in data:
        data[size].sort()

    average_data = defaultdict(list)
    for (size, polarization) in zip(average_cascade_per_round, average_polarization_per_round):
        average_data[size].append(polarization) 
    for size in average_data: 
        average_data[size].sort()
        
    return data, average_data
    


def parallel_cascade_experiment(corr, experiment_id, all_networks, number_of_iters):
    """Worker function for one cascade experiment."""
    before_network, after_network = all_networks([corr, experiment_id])

    before_data, average_before_data = create_data(number_of_iters, before_network)
    after_data, average_after_data = create_data(number_of_iters, after_network)

    largest_size_averaged = max(average_before_data.keys())
    largest_size = max(before_data.keys())
    if max(after_data.keys()) > largest_size:
        largest_size = max(after_data.keys())
    if max(average_after_data.keys()) > largest_size_averaged:
        largest_size_averaged = max(average_after_data.keys())

    return (before_data, after_data, largest_size), (average_before_data, average_after_data, largest_size_averaged)


def multiple_correlations_par(corr, all_networks):
    number_of_experiments = 10
    number_of_iters = 10000
    collection_of_all_before = defaultdict(list)
    collection_of_all_after = defaultdict(list)
    col_of_all_before_averaged = defaultdict(list)
    col_of_all_after_averaged = defaultdict(list)
    largest_size_of_all = 0
    largest_size_of_all_averaged = 0

    # Parallelize the experiments
    with ProcessPoolExecutor(max_workers=10) as executor:
        # Partial function for worker logic
        worker_function = partial(parallel_cascade_experiment, corr, all_networks=all_networks, number_of_iters=number_of_iters)
        
        # Execute in parallel
        results = list(executor.map(worker_function, range(number_of_experiments)))

    # Collect results
    for result in results:
        (before_data, after_data, largest_size), (average_before_data, average_after_data, largest_size_averaged) = result
        if largest_size > largest_size_of_all:
            largest_size_of_all = largest_size
        if largest_size_averaged > largest_size_of_all_averaged:
            largest_size_of_all_averaged = largest_size_averaged

        for size, polarizations in before_data.items():
            collection_of_all_before[size].extend(polarizations)
        for size, polarizations in after_data.items():
            collection_of_all_after[size].extend(polarizations)
        for size, polarizations in average_before_data.items():
            col_of_all_before_averaged[size].extend(polarizations)
        for size, polarizations in average_after_data.items():
            col_of_all_after_averaged[size].extend(polarizations)

    
    print("Finished all cascade experiments")
    return (
        (collection_of_all_before,
        collection_of_all_after),
        (col_of_all_before_averaged, col_of_all_after_averaged),
        (largest_size_of_all,
        largest_size_of_all_averaged),
    )