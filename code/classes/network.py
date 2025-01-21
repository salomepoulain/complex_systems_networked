import random
import numpy as np
from code.classes.node import Node

class Network:
    def __init__(self, num_nodes, correlation, starting_distribution, update_fraction):
        self.correlation = correlation 
        self.update_fraction = update_fraction

        self.nodesL = {Node(i, "L") for i in range(num_nodes * starting_distribution)}
        self.nodesR = {Node(i, "R") for i in range(num_nodes * (1 - starting_distribution))}

        self.all_nodes = list(self.nodesL) + list(self.nodesR)

        self.initialize_network()

    def initialize_network(self):
        """
        Initialize the network by connecting all nodes to a random number of other nodes.
        """
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # ik heb dit ff uit mn duim gezogen
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

        for node1 in self.all_nodes:
            for node2 in random.sample(self.all_nodes, random.randint(1, len(self.all_nodes))):
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
        active_nodes = [n for n in self.all_nodes if n.activation_state]
        active_node = random.choice(active_nodes)

        # If the node's behavior is inconsistent with its news source, break a tie and add a new connection
        if ((active_node.identity == 'L' and sL <= active_node.response_threshold) or
            (active_node.identity == 'R' and sR <= active_node.response_threshold)):
            
            # Break a tie with an active neighbor
            active_neighbors = [n for n in active_node.connections if n.activation_state]

            # Check if there are active neighbors to break ties with
            if active_neighbors:
                active_node.remove_edge(random.choice(active_neighbors))

            # Add a new random connection
            unconnected_nodes = [n for n in self.all_nodes if n != active_node and active_node not in n.connections]
            if unconnected_nodes:
                active_node.add_edge(random.choice(unconnected_nodes))

            ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
            # deze functie geeft nu geen zekerheid dat er ATIJD per round een nieuwe connectie wordt gemaakt, misschien moet er dus een loop # komen zodat dit wel elke ronde gebeurt
            ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 

    def update_round(self):
        """
        Perform a single update round.
        """
        sL, sR = self.generate_news_significance()

        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # nog niet duidelijk of deze fractie bij beiden identities even groot is, of wat de fractie grootte moet zijn
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # Select a fraction of nodes to become sampled
        [node.make_sampler() for node in random.sample(self.all_nodes, int(len(self.all_nodes) * self.update_fraction))]

        # Respond to the news intensities, continue this untill steady state is reached
        self.run_cascade(sL, sR)

        # Network adjustment
        self.network_adjustment(sL, sR)

        # Reset states for next round
        [node.reset_sampler() for node in self.all_nodes]

        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # Moet hierna ook alle activation states weer ge-reset worden?
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####     



        
