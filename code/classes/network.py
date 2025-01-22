import random
import numpy as np
import networkx as nx
from code.classes.node import Node
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter

class Network:
    def __init__(self, num_nodes, mean, correlation, starting_distribution, update_fraction, p=0.2, k=None):
        self.p = p
        self.k = k
        self.correlation = correlation 
        self.mean = mean
        self.update_fraction = update_fraction
        self.nodesL = {Node(i, "L") for i in range(int(num_nodes * starting_distribution))}
        self.nodesR = {Node(i + len(self.nodesL), "R") for i in range(int(num_nodes * (1 - starting_distribution)))}
        self.all_nodes = self.nodesL.union(self.nodesR)
        self.connections = set()
        self.graph = nx.Graph()

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
                    available_nodes.remove(node2)

            # Now use `p` to add random edges between any pair of nodes
            for node1 in self.all_nodes:
                for node2 in self.all_nodes:
                    if node1 != node2 and (node2 not in node1.connections):
                        if random.random() < self.p:
                            self.add_connection(node1, node2)
        else:
            # If no degree `k` is provided, fall back to the Erdős–Rényi model
            for node1 in self.all_nodes:
                for node2 in self.all_nodes:
                    if node1 != node2 and (node2 not in node1.connections):
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
            # print("\n")

    def network_adjustment(self, sL, sR):
        """
        Adjust the network by breaking ties and adding new connections.
        """
        # Select an active node involved in the cascade
        active_nodes = {n for n in self.all_nodes if n.activation_state}  # Set of active nodes
        # print("\n\n\n")
        # print(f"there are {len(active_nodes)} nodes active")

        if active_nodes:
            active_node = random.choice(list(active_nodes))

            # print(f"node {active_node.ID} is under investigation")
            # print(f"identity: {active_node.identity}")
            # print(f"Threshold: {active_node.response_threshold}")
            # print(f"sL:{sL} sR:{sR}")

            # Break ties if behavior is inconsistent with news source
            if ((active_node.identity == 'L' and sL <= active_node.response_threshold) or
                (active_node.identity == 'R' and sR <= active_node.response_threshold)):
                # print(f"node is {active_node.identity} and threshold is {active_node.response_threshold} which is higher than Left:{sL} Right{sR} ")
                
                # Break a tie with an active neighbor (use set for efficiency)
                active_neighbors = {n for n in active_node.connections if n.activation_state}
                # If active neighbors exist, remove an edge
                if active_neighbors:

                    # print(f"node has {len(list(active_neighbors))} active neighbors namely:")
                    # for i in active_neighbors:
                        # print(i.ID)
                    break_node = random.choice(list(active_neighbors))
                    self.remove_connection(active_node, break_node)
                    # print(f"removed edge from {active_node.ID} with {break_node.ID}")
                    
                    # only if an edge is removed, add an extra adge. 
                    node1 = random.choice(list(self.all_nodes))
                    node2 = random.choice(list(self.all_nodes))
                    # print(f"added connection from node: {node1.ID} to node: {node2.ID}")
                    self.add_connection(node1, node2)
            # else: 
            #     if active_node.identity == 'L' and sL > active_node.response_threshold:
                    # print(f"node is left and threshold is {active_node.response_threshold} which is lower than {sL}")
                # elif active_node.identity == 'R' and sR > active_node.response_threshold:
                    # print(f"node is right and threshold is {active_node.response_threshold} which is lower than {sR}")

                # Add a new random connection (ensure the node isn't already connected)
                
                # very inefficient line
                # unconnected_nodes = {n for n in self.all_nodes if n != active_node and active_node not in n.node_connections}
               
                # if unconnected_nodes:
                #     active_node.add_edge(random.choice(list(unconnected_nodes)))

            ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
            # deze functie geeft nu geen zekerheid dat er ATIJD per round een nieuwe connectie wordt gemaakt, misschien moet er dus een loop # komen zodat dit wel elke ronde gebeurt
            ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

    def update_round(self):
        """
        Perform a single update round.
        """
        sL, sR = self.generate_news_significance()
        # print(f"news significance left: {sL}")
        # print(f"news significance right: {sR}")
        # print("\n\n\n")


        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # nog niet duidelijk of deze fractie bij beiden identities even groot is, of wat de fractie grootte moet zijn
        ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### 
        # Select a fraction of nodes to become sampled

        for node in random.sample(list(self.all_nodes), int(len(self.all_nodes) * self.update_fraction)):
                # print(f"making node {node.ID} a sampler")
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

    def print(self):
        """
        Print network 
        """
        # create an empty graph
        self.graph = nx.Graph()

        # add all nodes
        for node in self.all_nodes:
            self.graph.add_node(node.ID, group = node.identity)

        # add edges
        for node in self.all_nodes:
            for connection in node.connections:
                self.graph.add_edges_from([(node.ID, connection.ID)])

        print(nx.average_clustering(self.graph))

        plt.figure(figsize=(8, 8))
        color_map = ["lightblue" if node[1]["group"] == "L" else "#FF6666" for node in self.graph.nodes(data=True)]
        # pos = nx.spring_layout(graph)
        pos = nx.kamada_kawai_layout(self.graph, scale=0.8)
        nx.draw(
            self.graph,
            pos,
            node_color=color_map,
            with_labels=False,
            edge_color="lightgray",  
            width=0.2,
            node_size=500,
            font_size=10,
        )
        # plt.title("Network Visualization")
        plt.show()

    def plot_degree_distribution(self):
        # calculate degrees of all nodes
        degrees = [deg for _, deg in self.graph.degree()]

        # count frequencies of each degree
        degree_counts = Counter(degrees)

        # sort by degree
        degrees, counts = zip(*sorted(degree_counts.items()))

        # plot the degree distribution
        plt.figure(figsize=(8, 6))
        plt.bar(degrees, counts, width=0.8, color="blue", edgecolor="black", alpha=0.7)
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

        





        

# num_nodes = 100
# correlation = 0.5
# update_fraction = 0.1
# starting_distribution = 0.5       # L / R ratio (niet per se nodig maar kan misschien leuk zijn om te varieern)
# p = 0.5
# k = 2

# network = Network(num_nodes, correlation, update_fraction, starting_distribution, p, k)

# for round in range(100):
#     network.update_round()