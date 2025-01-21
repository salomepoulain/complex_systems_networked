import random
import numpy as np
from code.classes.node import Node

class Network:
    def __init__(self, num_nodes, correlation, starting_distribution, update_fraction):
        self.correlation = correlation 
        self.update_fraction = update_fraction
        self.nodes = {Node(i, self.assign_identity(starting_distribution)) for i in range(num_nodes)}

        self.initialize_network()

    def assign_identity(self, starting_distribution):
        """
        Assign 'L' or 'R' identity to the node based on starting_distribution.
        starting_distribution = 1 means a 50/50 split.
        """
        return "L" if random.random() < 0.5 else "R"
        
    def initialize_network(self):
        """
        Initialize the network by connecting all nodes to a random number of other nodes.
        """
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # idk of dit de bedoeling is of we zo een netwerk willen
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        for node1 in self.nodes:
            for node2 in random.sample(self.nodes, random.randint(1, len(self.nodes))):
                self.add_connection(node1, node2)

    def add_connection(self, node1, node2):
        if node1 != node2:
            node1.add_edge(node2)
            node2.add_edge(node1)
            self.connections.add((node1, node2))

    def remove_connection(self, node1, node2):
        if node1 != node2:
            node1.remove_edge(node2)
            node2.remove_edge(node1)
            self.connections.discard((node1, node2))
        
    def generate_news_significance(self):
        """
        Generate news signifiance for both hubs based on their correlation.
        :return: Normalized signifiance (sL, sR) for the left and right media hubs.
        """
        covariance_matrix = [[1, self.correlation], [self.correlation, 1]]
        sL, sR = np.random.multivariate_normal(mean=[0, 0], cov=covariance_matrix)
        sL = (np.tanh(sL) + 1) / 2  
        sR = (np.tanh(sR) + 1) / 2
        return sL, sR
    
    def respond_until_steady_state(self, sL, sR):
        """
        Continue responding to the news intensities until a steady state is reached (no changes in activation state).
        """
        steady_state_reached = False
        while not steady_state_reached:
            steady_state_reached = True  
            for node in self.nodes:
                if node.identity == 'L' and node.respond(sL):
                    steady_state_reached = False 
                elif node.identity == 'R' and node.respond(sR):
                    steady_state_reached = False  

        [node.reset_sampler() for node in self.nodes]

    def network_adjustment(self, sL, sR):
        """
        Adjust the network by breaking ties and adding new connections.
        """
        # Select an active node involved in the cascade
        active_nodes = [n for n in self.nodes if n.activation_state]
        if not active_nodes: return
        
        node = random.choice(active_nodes)
        
        # If the node's behavior is inconsistent with its news source, break a tie and add a new connection
        if ((node.identity == 'L' and node.activation_state and sL <= node.response_threshold) or
            (node.identity == 'R' and node.activation_state and sR <= node.response_threshold)):
            
            # Break a tie with an active neighbor
            active_neighbors = [n for n in node.connections if n.activation_state]
            if active_neighbors: node.remove_edge(random.choice(active_neighbors))

            # Add a new random connection
            unconnected_nodes = [n for n in self.nodes if n != node and node not in n.connections]
            if unconnected_nodes: node.add_edge(random.choice(unconnected_nodes))

    def update_round(self):
        """
        Perform a single update round.
        """
        sL, sR = self.generate_news_significance()

        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # nog niet duidelijk of deze fractie bij beiden identities even groot is, of wat de fractie grootte moet zijn
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # Select a fraction of nodes to become sampled
        [node.make_sampler() for node in random.sample(self.nodes, int(len(self.nodes) * self.update_fraction))]

        # Respond to the news intensities, continue this untill steady state is reached
        self.respond_until_steady_state(sL, sR)

        # Network adjustment
        self.network_adjustment(sL, sR)


        
