import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt

path = 'muestras/M3/M30007.graphml'

G = nx.Graph(nx.read_graphml(path))
pos = nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), seed=10)

plt.figure()
nx.draw(G, pos)
plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
plt.show()

ohmic_ratio = 0.1

nodes = list(G.nodes())

for node in nodes:
    if np.random.rand() < ohmic_ratio:
        
        neighbors = list(G.neighbors(node))
        
        edges_to_add = itertools.combinations(neighbors, 2)
        
        G.add_edges_from(edges_to_add)
        
        G.remove_node(node)

plt.figure()
nx.draw(G, pos)
plt.xlim(-0.05, 0.05)
plt.ylim(-0.05, 0.05)
plt.show()