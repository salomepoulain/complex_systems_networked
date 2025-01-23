import numpy as np

class Node:
    def __init__(self, ID, identity, seed=None):
        self.ID = ID
        self.identity: str = identity
        self.seed = seed
        self.node_connections = set()
        self.response_threshold = self.set_response(seed)
        self.activation_state = False
        self.sampler_state = False

        self.cascade_size = 0
        self.cascade_left = 0
        self.last_of_cascade = False
        self.cascade_id = set()


    def set_response(self, seed):
        np.random.seed(seed)
        return np.random.random()
    
    def make_sampler(self):
        self.sampler_state = True

    def reset_sampler(self):
        self.sampler_state = False

    def reset_activation_state(self):
        self.activation_state = False

    def respond_sampler(self, intensity):

        new_activation_state = intensity > self.response_threshold

        assert self.activation_state==False, "activation state can't be true at this point"
        assert self.sampler_state==True, "node is not recognized as a sampler but treated as one"

        if new_activation_state: 
            self.activation_state = True
        


        
    def respond(self, intensity=0, inset = set(), analyze=False):
        """
        respond to the news intensity and returns False if the activation state did not change, True otherwise.
        """

        neighbors_activated = 0
        actually_activated = []
        new_activation_state = False

        if len(inset) > 0:
            if self.sampler_state:
                new_activation_state = intensity > self.response_threshold
                assert self in inset, "value should be contained in this set"
                
        else:
            if len(self.node_connections) > 0: 
                actually_activated = [node for node in self.node_connections if node.activation_state] 
                neighbors_activated = len(actually_activated)
                fraction_activated = neighbors_activated/len(self.node_connections)
            else:
                fraction_activated = 0
            new_activation_state = fraction_activated > self.response_threshold
                

        if not self.activation_state:
            if new_activation_state: 
                self.activation_state = True
            # assert new_activation_state == True, "The new state should only change to True, not to False"

            if analyze:
                self.last_of_cascade = True
                for neighbor in actually_activated:
                    neighbor.last_of_cascade = False
                    
                    self.cascade_size += neighbor.cascade_size
                    self.cascade_left += neighbor.cascade_left
                    self.cascade_id.update(neighbor.cascade_id)

                if self.ID == "L":
                    self.cascade_left +=1
                self.cascade_size +=1
                self.cascade_id.add(self.ID)
                assert self.cascade_left <= self.cascade_size, "fraction of left can't be larger than its size"
            return new_activation_state, self.node_connections.copy()

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

