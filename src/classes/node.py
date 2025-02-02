import numpy as np

class Node:
    """
    A node in the network, with a unique ID and a response threshold.
    The response threshold is a random number between 0 and 1, which is used to determine whether the node will respond to a piece of news.
    The node can be in one of two states: activated or not activated.
    The node can also be a sampler, which means that it will always respond to a piece of news, regardless of the response threshold.
    """
    def __init__(self, ID, identity, rng=None):
        """
        Initialize the node.

        Args:
            ID (int): The unique ID of the node.
            identity (str): The identity of the node (either "L" or "R").
            rng (np.random.Generator, optional): The random number generator to use. Defaults to None.
        
        Attributes:
            response_threshold (float): The response threshold of the node.
            sampler_state (bool): Whether the node is a sampler or not.
            activation_state (bool): Whether the node is activated or not.
            node_connections (set): The set of nodes that the node is connected to.
            cascade_size (int): The size of the cascade that the node is part of.
            cascade_left (int): The number of nodes left in the cascade that the node is part of.
            last_of_cascade (bool): Whether the node is the last node in the cascade that the node is part of.
            cascade_id (set): The set of nodes that are part of the cascade that the node is part of.  
        """
        self.ID = ID
        self.identity: str = identity
        self.node_connections = set()
        self.activation_state = False

        self.response_threshold = rng.random() if rng else np.random.random()
        self.sampler_state = False

        self.cascade_size = 0
        self.cascade_left = 0
        self.last_of_cascade = False
        self.cascade_id = set()
    
    def make_sampler(self):
        """
        Make the node a sampler, which means that it will always respond to a piece of news, regardless of the response threshold.
        """
        self.sampler_state = True

    def reset_sampler(self):
        """
        Reset the sampler state of the node.
        """
        self.sampler_state = False

    def reset_activation_state(self):
        self.activation_state = False
        
    def respond(self, intensity=0, analyze=False):
        """
        respond to the news intensity and returns False if the activation state did not change, True otherwise.

        Args:
            intensity (float): The intensity of the news.
            analyze (bool): Whether to analyze the cascade or not.
        
        Returns:
            bool: True if the activation state changed, False otherwise.
            set: The set of nodes that should be activated
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
        """
        Add an edge to the node.

        Args:
            node (Node): The node to add as an edge.
        """
        self.node_connections.add(node)

    def remove_edge(self, node):
        """
        Remove an edge from the node.

        Args:
            node (Node): The node to remove as an edge.
        """
        self.node_connections.discard(node)
    
    def reset_node(self):
        """
        Reset the node to its initial state.
        """
        self.cascade_size = 0
        self.cascade_left = 0
        self.last_of_cascade = False
        self.cascade_id = set()

    def respond_for_visuals(self, intensity):
        """
        respond to the news intensity and returns False if the activation state did not change, True otherwise.

        Args:
            intensity (float): The intensity of the news.
        """
        if self.sampler_state:
            new_activation_state = intensity > self.response_threshold
        else:
            if len(self.node_connections) != 0:
                fraction_activated = sum(1 for node in self.node_connections if node.activation_state) / len(self.node_connections)
                new_activation_state = fraction_activated > self.response_threshold
            else:
                fraction_activated = 0
                new_activation_state = False

        # return True if change was made else return False
        if new_activation_state != self.activation_state:
            self.activation_state = new_activation_state
            return True

        return False
        
    def __hash__(self):
        """
        Hash the node by its ID and identity.
        Needed for the set data structure.

        Returns:
            int: The hash of the node.
        """
        return hash((self.ID, self.identity)) 

    def __eq__(self, other):
        """
        Check if the node is equal to another node.
        """
        return isinstance(other, Node) and self.ID == other.ID and self.identity == other.identity

