import random

class Node:
    def __init__(self, ID, identity):
        self.ID = ID
        self.identity: str = identity

        self.node_connections = set()
        self.response_threshold = random.uniform(0, 1)
        self.activation_state = False
        self.sampler_state = False

    def make_sampler(self):
        self.sampler_state = True

    def reset_sampler(self):
        self.sampler_state = False
        
    def respond(self, intensity):
        """
        respond to the news intensity and returns False if the activation state did not change, True otherwise.
        """
        if self.sampler_state:
            new_activation_state = intensity > self.response_threshold
        else:
            fraction_activated = sum(1 for node in self.connections if node.activation_state) / len(self.connections)
            new_activation_state = fraction_activated > self.response_threshold

        if new_activation_state != self.activation_state:
            self.activation_state = new_activation_state
            return True

        return False

    def add_edge(self, node):
        self.node_connections.add(node)

    def remove_edge(self, node):
        self.node_connections.discard(node)
