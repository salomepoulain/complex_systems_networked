import random
import numpy as np
from src.classes.node import Node
from scipy import stats

class Network:
    def __init__(self, num_nodes, mean=0, correlation=-1, starting_distribution=0.5, update_fraction=0.3, p=0.1, k=None):
        self.p = p
        self.k = k
        self.correlation = correlation 
        self.mean = mean
        self.alterations = 0
        self.update_fraction = update_fraction
        self.nodesL = {Node(i, "L") for i in range(int(num_nodes * starting_distribution))}
        self.nodesR = {Node(i + len(self.nodesL), "R") for i in range(int(num_nodes * (1 - starting_distribution)))}
        self.connections = set()
        self.all_nodes = self.nodesL.union(self.nodesR)

        self.initialize_random_network()

    def initialize_random_network(self):
        """
        Initialize the network by connecting all nodes with a probability `p`.
        If `p` is very low, the network will resemble a regular network with fixed degree `k`.
        If `p` is high, it will resemble an Erdős–Rényi random network.
        """
        if self.k is not None:
            # If degree `k` is provided, ensure each node has exactly `k` connections.
            # This creates a regular network first, and then we adjust using `p`.
            for node1 in self.all_nodes:
                # Create k regular connections for each node
                available_nodes = list(self.all_nodes - {node1})
                for _ in range(self.k):
                    node2 = random.choice(available_nodes)
                    self.add_connection(node1, node2)
                    # self.add_connection(node2, node1)
                    available_nodes.remove(node2)

            # Now use `p` to add random edges between any pair of nodes
            for node1 in self.all_nodes:
                for node2 in self.all_nodes:
                    if node1 != node2 and (node2 not in node1.node_connections):
                        if random.random() < self.p:
                            self.add_connection(node1, node2)
        else:
            # If no degree `k` is provided, fall back to the Erdős–Rényi model
            for node1 in self.all_nodes:
                for node2 in self.all_nodes:
                    if node1 != node2 and (node2 not in node1.node_connections):
                        if random.random() < self.p:
                            self.add_connection(node1, node2)


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
        stims = np.random.multivariate_normal(mean = [self.mean, self.mean], cov = covar, size = 1)
        stims_perc = stats.norm.cdf(stims, loc = 0, scale = 1) 
        return stims_perc[0][0], stims_perc[0][1]
    
    def run_cascade(self, sL, sR):
        """
        Continue responding to the news intensities until a steady state is reached (no changes in activation state).
        This is the cascade event.
        """
        steady_state_reached = False
        while not steady_state_reached:
            steady_state_reached = True  
            for nodeL in self.nodesL:
                if nodeL.respond(sL):
                    steady_state_reached = False
            for nodeR in self.nodesR:
                if nodeR.respond(sR):
                    steady_state_reached = False

    def network_adjustment(self, sL, sR):
        """
        Adjust the network by breaking ties and adding new connections.
        """
        # Select an active node involved in the cascade
        active_nodes = {n for n in self.all_nodes if n.activation_state}  # Set of active nodes
        if active_nodes:
            active_node = random.choice(list(active_nodes))

            # Break ties if behavior is inconsistent with news source
            if ((active_node.identity == 'L' and sL <= active_node.response_threshold) or
                (active_node.identity == 'R' and sR <= active_node.response_threshold)):
                
                # Break a tie with an active neighbor (use set for efficiency)
                active_neighbors = {n for n in active_node.node_connections if n.activation_state}

                number_of_connections = len(self.connections)
                # If active neighbors exist, remove an edge
                if active_neighbors:
                    self.alterations +=1
                    # print(f"removed edge from {active_node.ID}")
                    break_node = random.choice(list(active_neighbors))
                    self.remove_connection(active_node, break_node)
                    
                    # only if an edge is removed, add an extra adge. 
                    node1 = random.choice(list(self.all_nodes))
                    cant_be_picked = node1.node_connections.copy()
                    cant_be_picked.add(node1)
                    node2 = random.choice(list(self.all_nodes - cant_be_picked))

                    # print(f"added connection from node: {node1.ID} to node: {node2.ID}")
                    self.add_connection(node1, node2)
                
                assert number_of_connections == len(self.connections), "invalid operation took place, new number of edges is different than old"


    def update_round(self):
        """
        Perform a single update round.
        """
        self.alterations = 0
        sL, sR = self.generate_news_significance()

        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # nog niet duidelijk of deze fractie bij beiden identities even groot is, of wat de fractie grootte moet zijn
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # Select a fraction of nodes to become sampled
        for node in random.sample(list(self.all_nodes), int(len(self.all_nodes) * self.update_fraction)):
            node.make_sampler()

        # Respond to the news intensities, continue this untill steady state is reached
        self.run_cascade(sL, sR)

        # Network adjustment
        self.network_adjustment(sL, sR)

        # Reset states for next round
        for node in self.all_nodes:
            node.reset_sampler()
            node.reset_activation_state()

        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # Moet hierna ook alle activation states weer ge-reset worden?
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####     



        

# num_nodes = 100
# correlation = 0.5
# update_fraction = 0.1
# starting_distribution = 0.5       # L / R ratio (niet per se nodig maar kan misschien leuk zijn om te varieern)
# p = 0.5
# k = 2

# network = Network(num_nodes, correlation, update_fraction, starting_distribution, p, k)

# for round in range(100):
#     network.update_round()