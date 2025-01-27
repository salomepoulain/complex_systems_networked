import random
import numpy as np
from src.classes.node import Node
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
# from powerlaw import Fit

class Network:
    def __init__(self, network, num_nodes, mean=0, correlation=-1, starting_distribution=0.5, update_fraction=0.2, seed=None, p=0.1, k=None, m=2):
        self.p = p
        self.k = k
        self.m = m
        self.seed = seed
        self.correlation = correlation 
        self.mean = mean
        self.activated = set()
        self.alterations = 0
        self.update_fraction = update_fraction

        self.nodesL = {Node(i, "L", seed=i*3) for i in range(int(num_nodes * starting_distribution))}
        self.nodesR = {Node(i + len(self.nodesL), "R", seed=i+num_nodes*2) for i in range(int(num_nodes * (1 - starting_distribution)))}
        self.connections = set()
        self.all_nodes = self.nodesL.union(self.nodesR)

        # for visuals
        self.iterations = 0
        self.removed_edge = []
        self.new_edge = []
        
        if network == 'scale_free':
            self.initialize_scale_free_network()
        elif network == 'random':
            self.initialize_random_network()
        else:
            assert network is None, "Invalid network type provided. Use 'scale_free' or 'random'."
        

    def initialize_random_network(self):
        """
        Initialize the network by connecting all nodes with a probability `p`.
        If `p` is very low, the network will resemble a regular network with fixed degree `k`.
        If `p` is high, it will resemble an Erdős–Rényi random network.
        """

        np.random.seed(self.seed)
        if self.seed != None:
            self.seed+=1
        if self.k is not None:
            print(f"A Wattz-Strogatz network is initialized with beta value {self.p} and regular network degree {self.k}")
            # If degree `k` is provided, ensure each node has exactly `k` connections.
            # This creates a regular network first, and then we adjust using `p`.
            for node1 in self.all_nodes:
                # Create k regular connections for each node
                available_nodes = list(self.all_nodes - {node1})
                for _ in range(self.k):
                    node2 = np.random.choice(available_nodes)
                    self.add_connection(node1, node2)
                    available_nodes.remove(node2)

            # Now use `p` to add random edges between any pair of nodes
            for node1 in self.all_nodes:
                for node2 in self.all_nodes:
                    if node1 != node2 and (node2 not in node1.node_connections):
                        if np.random.random() < self.p:
                            self.add_connection(node1, node2)
        else:
            print(f'A random network is initialized with p: {self.p} and {len(self.all_nodes)} nodes')
            # If no degree `k` is provided, fall back to the Erdős–Rényi model
            for node1 in self.all_nodes:
                for node2 in self.all_nodes:
                    if node1 != node2 and (node2 not in node1.node_connections):
                        if np.random.random() < self.p:
                            self.add_connection(node1, node2)

    def initialize_scale_free_network(self):
        assert self.m < len(self.all_nodes), "Number of connections `m` must be less than the number of nodes."
        assert self.m > 0, "Number of connections `m` must be greater than 0."

        # Create a list of nodes to work with
        all_nodes_list = list(self.all_nodes)

        # Select initial m nodes and fully connect them
        m0_nodes = random.sample(all_nodes_list, self.m)
        for i in range(len(m0_nodes)):
            for j in range(i + 1, len(m0_nodes)):
                self.add_connection(m0_nodes[i], m0_nodes[j])

        # Track degrees for preferential attachment
        degrees = {node: len(node.node_connections) for node in self.all_nodes if node in m0_nodes}

        # Add remaining nodes using preferential attachment
        remaining_nodes = list(set(all_nodes_list) - set(m0_nodes))

        for new_node in remaining_nodes:
            # Calculate cumulative degree distribution for preferential attachment
            total_degree = sum(degrees.values())
            cumulative_probabilities = []
            cumulative_sum = 0

            connection_candidates = list(degrees.keys())
            for node in connection_candidates:
                cumulative_sum += degrees[node] / total_degree
                cumulative_probabilities.append(cumulative_sum)

            # Select nodes to connect to, with probability proportional to their degree
            connected_nodes = set()
            while len(connected_nodes) < self.m:
                r = random.random()
                for idx, cumulative_prob in enumerate(cumulative_probabilities):
                    if r <= cumulative_prob:
                        connected_nodes.add(connection_candidates[idx])
                        break

            # Add connections
            for target_node in connected_nodes:
                self.add_connection(new_node, target_node)
                degrees[target_node] += 1  # Update degree for target node

            degrees[new_node] = self.m  # New node has `m` connections

        # Verify scale-free properties and optionally plot
        self.verify_scale_free_distribution(plot=True)

    def verify_scale_free_distribution(self, plot):
        """
        Check if the network exhibits scale-free characteristics
        """
        # Calculate node degrees
        degrees = [len(node.node_connections) for node in self.all_nodes]
        
        # Compute log-log plot for degree distribution
        degree_counts = {}
        for degree in degrees:
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        
        unique_degrees = list(degree_counts.keys())
        frequencies = list(degree_counts.values())
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.loglog(unique_degrees, frequencies, 'bo')
            plt.title('Degree Distribution (Log-Log Scale)')
            plt.xlabel('Degree')
            plt.ylabel('Frequency')
            plt.show()
        
        # Basic scale-free network indicators
        assert max(degrees) > np.mean(degrees) * 2, "Network lacks high-degree nodes"
        assert len([d for d in degrees if d > np.mean(degrees) * 2]) > 0, "No significant hub nodes"
        fit = Fit(degrees)
        print("Power-law alpha:", fit.power_law.alpha)
        print("Goodness of fit (p-value):", fit.power_law.KS())

    def add_connection(self, node1, node2):
        """Add an undirected connection between two nodes (if not already present)."""
        if node1 != node2: 
            node1.add_edge(node2)
            node2.add_edge(node1)
            self.connections.add((node1, node2))
            self.connections.add((node2, node1))

    def remove_connection(self, node1, node2):
        """Remove the connection between two nodes if it exists."""
        if node1 != node2:
            node1.remove_edge(node2)
            node2.remove_edge(node1)
            self.connections.remove((node1, node2))
            self.connections.remove((node2, node1))
        
    def generate_news_significance(self):
        """
        Generate news signifiance for both hubs based on their correlation.
        :return: Normalized signifiance (sL, sR) for the left and right media hubs.
        """
        covar = [[1, self.correlation ], [self.correlation, 1]]
        np.random.seed(self.seed)
        stims = np.random.multivariate_normal(mean = [self.mean, self.mean], cov = covar, size = 1)
        stims_perc = stats.norm.cdf(stims, loc = 0, scale = 1) 
        return stims_perc[0][0], stims_perc[0][1]

    def run_cascade(self, sL, sR, all_samplers):
        """
        Continue responding to the news intensities until a steady state is reached (no changes in activation state).
        This is the cascade event.
        """
        self.activated=set()
        steady_state_reached = True
        union_to_consider= set()
        all_left, all_right = all_samplers

        # all_players = all_left.union(all_right)

        # inject news for left oriented nodes
        for nodeL in all_left:
            nodeL.reset_node()
            active_state, to_consider_L = nodeL.respond(sL)
            if active_state:
                union_to_consider.update(to_consider_L)
                steady_state_reached = False
                self.activated.add(nodeL)

        # inject news for right oriented nodes
        for nodeR in all_right:
            nodeR.reset_node()
            active_state, to_consider_R = nodeR.respond(sR)
            if active_state:
                union_to_consider.update(to_consider_R)
                steady_state_reached = False
                self.activated.add(nodeR)

        while not steady_state_reached:
            steady_state_reached = True
            new_to_consider = set()

            for individual in union_to_consider:
                # omit redundant checks by returning only the neighbors of newly activated nodes. 
                active_state, to_consider = individual.respond()
                if active_state:
                    steady_state_reached=False
                    self.activated.add(individual)
                    new_to_consider.update(to_consider)
            union_to_consider = new_to_consider

    def analyze_network(self, sL, sR):
        pass

    def network_adjustment(self, sL, sR):
        """
        Adjust the network by breaking ties and adding new connections.
        """
        self.new_edge = []
        self.removed_edge = []
        # Select an active node involved in the cascade
        np.random.seed(self.seed)

        # can maybe be done more efficiently if done dynamically
        # active_nodes = {n for n in self.all_nodes if n.activation_state}  # Set of active nodes

        if len(self.activated) >0:
            active_node = np.random.choice(list(self.activated))

            if ((active_node.identity == 'L' and sL <= active_node.response_threshold) or
                (active_node.identity == 'R' and sR <= active_node.response_threshold)):
                
                # Break a tie with an active neighbor (use set for efficiency)
                active_neighbors = {n for n in active_node.node_connections if n.activation_state}
                number_of_connections = len(self.connections)

                # If active neighbors exist, remove an edge
                if len(active_neighbors) > 0:
                    
                    self.alterations+=1
                    
                    # remove edge
                    break_node = np.random.choice(list(active_neighbors))
                    self.remove_connection(active_node, break_node)
                    self.removed_edge.extend([active_node.ID, break_node.ID])
                    
                    # only if an edge is removed, add an extra adge. 
                    node1 = np.random.choice(list(self.all_nodes))
                    cant_be_picked = node1.node_connections.copy()
                    cant_be_picked.add(node1)
                    node2 = np.random.choice(list(self.all_nodes - cant_be_picked))
                    self.new_edge.extend([node1.ID, node2.ID])

                    # add edge
                    self.add_connection(node1, node2)
                
                assert number_of_connections == len(self.connections), "invalid operation took place, new number of edges is different than old"


    def update_round(self):
        """
        Perform a single update round.
        """
        self.iterations += 1

        if self.seed != None:
            self.seed+=1
        sL, sR = self.generate_news_significance()

        np.random.seed(self.seed)
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # nog niet duidelijk of deze fractie bij beiden identities even groot is, of wat de fractie grootte moet zijn
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # Select a fraction of nodes to become sampled
        all_samplers_L, all_samplers_R = set(), set()
        for node in np.random.choice(list(self.all_nodes), int(len(self.all_nodes) * self.update_fraction), replace=False):
            if node.identity == 'L':
                all_samplers_L.add(node)
            elif node.identity == 'R':
                all_samplers_R.add(node)
            else:
                raise ValueError("node identity should be assigned")
            assert node.sampler_state == False, "at this point all samplers states should be false"
            assert node.activation_state == False, "at this point all nodes should be inactive"
            node.sampler_state = True

        # Respond to the news intensities, continue this untill steady state is reached
        self.run_cascade(sL, sR, (all_samplers_L, all_samplers_R))

        # Network adjustment
        self.network_adjustment(sL, sR)

        # Reset states for next round
        for node in self.activated:
            node.reset_activation_state()
            
        self.activated = set()

    def give_alterations(self):
        return self.alterations
