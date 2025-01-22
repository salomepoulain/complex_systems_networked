from code.classes.network import Network
from code.classes.node import Node
from code.smallvisual import plot_network

def create_network():
    netwerkje = Network(10, -1)
    plot_network(netwerkje)
create_network()
