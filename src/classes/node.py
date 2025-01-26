import numpy as np
from collections import OrderedDict

class Node:
    def __init__(self, ID, identity, rng=None):
        self.ID = ID
        self.identity: str = identity
        self.node_connections = set()
        # self.response_threshold = self.set_response(seed)
        self.activation_state = False

        self.response_threshold = rng.random() if rng else np.random.random()
        self.sampler_state = False

        self.cascade_size = 0
        self.cascade_left = 0
        self.last_of_cascade = False
        self.cascade_id = set()
    
    def make_sampler(self):
        self.sampler_state = True

    def reset_sampler(self):
        self.sampler_state = False

    def reset_activation_state(self):
        self.activation_state = False

        
    def respond(self, intensity=0, analyze=False):
        """
        respond to the news intensity and returns False if the activation state did not change, True otherwise.
        """

        neighbors_activated = 0
        actually_activated = []
        new_activation_state = False

        # if len(inset) > 0:
        if self.sampler_state:
            new_activation_state = intensity > self.response_threshold
            self.sampler_state = False
            # assert self in inset, "value should be contained in this set"
                
        else:
            if len(self.node_connections) > 0: 
                actually_activated = [node for node in self.node_connections if node.activation_state] 
                neighbors_activated = len(actually_activated)
                fraction_activated = neighbors_activated/len(self.node_connections)
            else:
                fraction_activated = 0
            new_activation_state = fraction_activated > self.response_threshold

        # assert self.activation_state == False, "only nodes should be selected that are inactive at this point"
        if not self.activation_state:
            nodes_to_return = set()
            if new_activation_state: 
                self.activation_state = True
            # assert new_activation_state == True, "The new state should only change to True, not to False"
            
                if analyze:
                    self.last_of_cascade = True
                    if self.identity == "L":
                        self.cascade_left +=1
                    if self.identity =='L':
                        waarde = 1
                    else:
                        waarde = -1
                    for neighbor in actually_activated:
                        neighbor.last_of_cascade = False
                        
                        self.cascade_size += neighbor.cascade_size
                        self.cascade_left += neighbor.cascade_left
                        # assert (int(self.ID), int(waarde)) not in neighbor.cascade_id, "this node is already part of cascade"
                        self.cascade_id.update(neighbor.cascade_id)
                    
                    self.cascade_size +=1
                    self.cascade_id.add((int(self.ID), int(waarde)))

                    assert self.cascade_left <= self.cascade_size, "fraction of left can't be larger than its size"

                nodes_to_return = {n for n in self.node_connections if not n.activation_state}
            return new_activation_state, nodes_to_return

        return False, set()

    def add_edge(self, node):
        self.node_connections.add(node)

    def remove_edge(self, node):
        self.node_connections.discard(node)
    
    def reset_node(self):
        # self.activation_state = False
        # self.sampler_state = False
        self.cascade_size = 0
        self.cascade_left = 0
        self.last_of_cascade = False
        self.cascade_id = set()
        

    def __hash__(self):
        # Needed for set operations to work correctly
        return hash((self.ID, self.identity)) 

    def __eq__(self, other):
        return isinstance(other, Node) and self.ID == other.ID and self.identity == other.identity

