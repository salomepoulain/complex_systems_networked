import random

class Node:
    def __init__(self, ID, identity):
        self.ID = ID
        self.identity: str = identity

        self.connections = set()
        self.response_threshold = random.uniform(0, 1)
        self.activation_state = False
        self.sampler_state = False

    def make_sampler(self):
        self.sampler_state = True

    def reset_sampler(self):
        self.sampler_state = False

    def reset_activation_state(self):
        self.activation_state = False
        
    def respond(self, intensity):
        """
        respond to the news intensity and returns False if the activation state did not change, True otherwise.
        """
        if self.sampler_state:
            new_activation_state = intensity > self.response_threshold
        else:
            if len(self.connections) > 0:
                fraction_activated = sum(1 for node in self.connections if node.activation_state) / len(self.connections)
            else:
                fraction_activated = 0
            new_activation_state = fraction_activated > self.response_threshold

        if new_activation_state != self.activation_state:
            self.activation_state = new_activation_state
            return True

        return False

    def add_edge(self, node):
        self.connections.add(node)

    def remove_edge(self, node):
        self.connections.discard(node)

    def __hash__(self):
        # Needed for set operations to work correctly
        return hash((self.ID, self.identity)) 

    def __eq__(self, other):
        return isinstance(other, Node) and self.ID == other.ID and self.identity == other.identity

